"""Persist Chroma chunks with an SQLite registry for document-level visibility."""

import sqlite3
from contextlib import closing
from pathlib import Path
from threading import RLock

import chromadb

from app.core.errors import AppError
from app.rag.chunker import Chunk
from app.schemas.chat import Source
from app.schemas.document import Document


class VectorStore:
    def __init__(self, path: Path, name: str, embedding_model: str) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.registry = path / f"{name}.documents.sqlite3"
        self.lock = RLock()
        self.client = chromadb.PersistentClient(path=str(path.resolve()))
        self.collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=None,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"embedding_model": embedding_model},
        )
        if (self.collection.metadata or {}).get("embedding_model") != embedding_model:
            raise RuntimeError("Use a new COLLECTION_NAME when changing EMBEDDING_MODEL.")
        with closing(self._connect()) as db, db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS documents (id TEXT PRIMARY KEY, payload TEXT)"
            )
            pending = db.execute("SELECT id FROM documents WHERE payload IS NULL").fetchall()
        for (document_id,) in pending:
            self._purge(document_id)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.registry)

    def exists(self, document_id: str) -> bool:
        with self.lock, closing(self._connect()) as db:
            return db.execute(
                "SELECT 1 FROM documents WHERE id = ? AND payload IS NOT NULL",
                (document_id,),
            ).fetchone() is not None

    def has_documents(self) -> bool:
        with self.lock, closing(self._connect()) as db:
            return db.execute(
                "SELECT 1 FROM documents WHERE payload IS NOT NULL LIMIT 1"
            ).fetchone() is not None

    def list_documents(self) -> list[Document]:
        with self.lock, closing(self._connect()) as db:
            rows = db.execute("SELECT payload FROM documents WHERE payload IS NOT NULL").fetchall()
        return sorted(
            (Document.model_validate_json(payload) for (payload,) in rows),
            key=lambda document: document.indexed_at,
            reverse=True,
        )

    def insert(self, document: Document, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Publish a document after all batches are durable. A null row marks recoverable work."""
        key = document.document_id
        if not chunks or len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one embedding.")
        with self.lock, closing(self._connect()) as db:
            if self.exists(key):
                raise AppError(409, "These file contents are already indexed.")
            self._purge(key)
            with db:
                db.execute("INSERT INTO documents (id) VALUES (?)", (key,))
            try:
                size = min(self.client.get_max_batch_size(), 256)
                for start in range(0, len(chunks), size):
                    batch = chunks[start:start + size]
                    self.collection.add(
                        ids=[chunk.chunk_id for chunk in batch],
                        documents=[chunk.text for chunk in batch],
                        embeddings=vectors[start:start + size],
                        metadatas=[
                            {
                                "document_id": key,
                                "filename": document.filename,
                                "page": chunk.page or 0,
                                "index": chunk.index,
                            }
                            for chunk in batch
                        ],
                    )
                with db:
                    db.execute("UPDATE documents SET payload = ? WHERE id = ?", (document.model_dump_json(), key))
            except Exception:
                self._purge(key)
                raise

    def _purge(self, document_id: str) -> None:
        with closing(self._connect()) as db, db:
            pending = db.execute(
                "SELECT 1 FROM documents WHERE id = ? AND payload IS NULL", (document_id,)
            ).fetchone()
            if pending:
                self.collection.delete(where={"document_id": document_id})
                db.execute("DELETE FROM documents WHERE id = ?", (document_id,))

    def delete(self, document_id: str) -> None:
        with self.lock, closing(self._connect()) as db:
            if not self.exists(document_id):
                self._purge(document_id)
                raise AppError(404, "Document not found.")
            with db:
                db.execute("UPDATE documents SET payload = NULL WHERE id = ?", (document_id,))
            self._purge(document_id)

    def search(self, vector: list[float], top_k: int, max_distance: float) -> list[Source]:
        with self.lock, closing(self._connect()) as db:
            rows = db.execute("SELECT id FROM documents WHERE payload IS NOT NULL").fetchall()
            if not rows:
                return []
            result = self.collection.query(
                query_embeddings=[vector],
                n_results=top_k,
                where={"document_id": {"$in": [key for (key,) in rows]}},
                include=["documents", "metadatas", "distances"],
            )
        sources = []
        for chunk_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0],
            result["distances"][0], strict=True,
        ):
            distance = min(2.0, max(0.0, distance))
            if distance <= max_distance:
                sources.append(Source(
                    citation=f"S{len(sources) + 1}",
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    page=meta["page"] or None,
                    chunk_id=chunk_id,
                    distance=distance,
                    excerpt=text,
                ))
        return sources
