"""Validate, embed, and index uploaded documents."""

import asyncio
import hashlib
import logging
from datetime import datetime, timezone

from starlette.concurrency import run_in_threadpool

from app.core.errors import AppError
from app.rag.chunker import Chunker
from app.rag.embeddings import Embedder
from app.rag.loader import load_document, validate_filename
from app.rag.vector_store import VectorStore
from app.schemas.document import Document

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(
        self, store: VectorStore, chunker: Chunker, embedder: Embedder, max_bytes: int
    ) -> None:
        self.store = store
        self.chunker = chunker
        self.embedder = embedder
        self.max_bytes = max_bytes
        self.upload_lock = asyncio.Lock()

    async def ingest(self, filename: str, content: bytes) -> Document:
        filename = validate_filename(filename)
        if len(content) > self.max_bytes:
            raise AppError(413, "The file exceeds the upload size limit.")
        if not content:
            raise AppError(422, "The uploaded file is empty.")
        document_id = await run_in_threadpool(lambda: hashlib.sha256(content).hexdigest())
        async with self.upload_lock:
            if await run_in_threadpool(self.store.exists, document_id):
                raise AppError(409, "These file contents are already indexed.")
            pages = await run_in_threadpool(load_document, filename, content)
            chunks = await run_in_threadpool(self.chunker.split, document_id, pages)
            vectors = await self.embedder.embed([chunk.text for chunk in chunks])
            document = Document(
                document_id=document_id,
                filename=filename,
                size_bytes=len(content),
                chunk_count=len(chunks),
                indexed_at=datetime.now(timezone.utc),
            )
            await run_in_threadpool(self.store.insert, document, chunks, vectors)
        logger.info("document_indexed", extra={"document_id": document_id, "chunk_count": len(chunks)})
        return document
