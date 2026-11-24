"""Find relevant source excerpts for a standalone question."""

from starlette.concurrency import run_in_threadpool

from app.rag.embeddings import Embedder
from app.rag.vector_store import VectorStore
from app.schemas.chat import Source


class Retriever:
    def __init__(self, store: VectorStore, embedder: Embedder, top_k: int, max_distance: float) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.max_distance = max_distance

    async def has_documents(self) -> bool:
        return await run_in_threadpool(self.store.has_documents)

    async def retrieve(self, query: str) -> list[Source]:
        vector, = await self.embedder.embed([query])
        return await run_in_threadpool(self.store.search, vector, self.top_k, self.max_distance)
