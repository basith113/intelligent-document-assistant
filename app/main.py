"""FastAPI application and shared resource lifecycle."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import AsyncOpenAI
from starlette.concurrency import run_in_threadpool

from app.api import chat, documents
from app.core.config import Settings
from app.core.errors import AppError
from app.core.logging import configure_logging
from app.rag.chunker import Chunker
from app.rag.embeddings import Embedder
from app.rag.generator import Generator
from app.rag.pipeline import Pipeline
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore
from app.services.chat_service import ChatService
from app.services.document_service import DocumentService

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, openai_client: AsyncOpenAI | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        configure_logging()
        config = settings or Settings()
        key = config.openai_api_key.get_secret_value().strip()
        client = openai_client
        if client is None and key:
            client = AsyncOpenAI(api_key=key, timeout=60.0, max_retries=2)
        try:
            store = await run_in_threadpool(
                VectorStore, config.chroma_path, config.collection_name, config.embedding_model
            )
            chunker = await run_in_threadpool(
                Chunker, config.chunk_size, config.chunk_overlap, config.embedding_model
            )
            embedder = Embedder(client, config.embedding_model)
            retriever = Retriever(store, embedder, config.top_k, config.max_distance)
            generator = Generator(client, config.openai_model, config.temperature)
            application.state.documents = DocumentService(store, chunker, embedder, config.max_upload_bytes)
            application.state.chat = ChatService(
                Pipeline(retriever, generator), config.max_history_messages, config.max_sessions
            )
            logger.info("api_ready")
            yield
        finally:
            if client is not None:
                await client.close()

    application = FastAPI(title="Knowledge AI Assistant", version="1.0.0", lifespan=lifespan)
    application.include_router(documents.router)
    application.include_router(chat.router)

    @application.exception_handler(AppError)
    async def app_error(_: Request, error: AppError) -> JSONResponse:
        if error.status >= 500 or error.status == 429:
            logger.warning("request_failed", extra={"error_type": type(error.__cause__ or error).__name__})
        return JSONResponse({"detail": error.detail}, status_code=error.status)

    @application.exception_handler(Exception)
    async def unexpected_error(_: Request, error: Exception) -> JSONResponse:
        logger.error("unexpected_request_failure", extra={"error_type": type(error).__name__})
        return JSONResponse({"detail": "The server could not complete the request."}, status_code=500)

    @application.get("/health", tags=["status"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return application


app = create_app()
