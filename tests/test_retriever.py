"""Test real Chroma storage and document lifecycle with deterministic vectors."""

import asyncio
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.rag.chunker import Chunk
from app.rag.embeddings import Embedder
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.schemas.document import Document

MODEL = "text-embedding-3-small"


def entry(key):
    return Document(
        document_id=key, filename=f"{key}.md", size_bytes=16,
        chunk_count=1, indexed_at=datetime.now(timezone.utc),
    )


def test_embedding_batch_size_and_input_order():
    def response(input, model):
        data = [
            SimpleNamespace(index=index, embedding=[float(value)])
            for index, value in reversed(list(enumerate(input)))
        ]
        return SimpleNamespace(data=data)

    create = AsyncMock(side_effect=response)
    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    inputs = [str(index) for index in range(130)]
    assert asyncio.run(Embedder(client, MODEL).embed(inputs)) == [[float(i)] for i in range(130)]
    assert [len(call.kwargs["input"]) for call in create.await_args_list] == [64, 64, 2]


def test_retrieval_deletion_and_persistence_after_process_restart(tmp_path):
    store = VectorStore(tmp_path, "knowledge_base", MODEL)
    store.insert(entry("policy"), [Chunk("policy:0", "Annual leave", 3, 0)], [[1.0, 0.0]])
    store.insert(entry("catalogue"), [Chunk("catalogue:0", "Product list", None, 0)], [[0.0, 1.0]])
    embedder = AsyncMock(spec=Embedder)
    embedder.embed.return_value = [[1.0, 0.0]]
    retriever = Retriever(store, embedder, 5, 0.2)
    sources = asyncio.run(retriever.retrieve("How much leave?"))
    assert [(source.filename, source.page, source.chunk_id) for source in sources] == [
        ("policy.md", 3, "policy:0")
    ]
    assert sources[0].distance == pytest.approx(0)
    assert len(store.search([1.0, 0.0], 1, 2.0)) == 1
    store.delete("catalogue")
    script = """
import sys
from pathlib import Path
from app.rag.vector_store import VectorStore
store = VectorStore(Path(sys.argv[1]), 'knowledge_base', 'text-embedding-3-small')
assert [document.document_id for document in store.list_documents()] == ['policy']
assert store.search([1.0, 0.0], 5, 2.0)[0].chunk_id == 'policy:0'
"""
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, timeout=45)


def test_failed_storage_write_can_be_retried(tmp_path, monkeypatch):
    store = VectorStore(tmp_path, "knowledge_base", MODEL)
    add = store.collection.add

    def fail_after_write(**kwargs):
        add(**kwargs)
        raise RuntimeError("simulated database failure")

    chunks = [Chunk("policy:0", "Annual leave", 1, 0)]
    with monkeypatch.context() as patch:
        patch.setattr(store.collection, "add", fail_after_write)
        with pytest.raises(RuntimeError, match="simulated database failure"):
            store.insert(entry("policy"), chunks, [[1.0, 0.0]])
    assert not store.list_documents()
    assert store.collection.count() == 0
    store.insert(entry("policy"), chunks, [[1.0, 0.0]])
    assert store.collection.count() == 1


def test_model_change_requires_new_collection(tmp_path):
    VectorStore(tmp_path, "knowledge_base", MODEL)
    with pytest.raises(RuntimeError, match="COLLECTION_NAME"):
        VectorStore(tmp_path, "knowledge_base", "other-model")
