"""Answer questions over JSON or an SSE response stream."""

import json
import logging
from collections.abc import AsyncGenerator
from contextlib import aclosing
from uuid import UUID

from anyio import CancelScope
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse
from starlette.types import Send

from app.core.errors import AppError
from app.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(prefix="/chat", tags=["chat"])
logger = logging.getLogger(__name__)


class EventStreamResponse(StreamingResponse):
    """Close the generator even if a client disconnects during a send."""

    def __init__(self, events: AsyncGenerator[str, None]) -> None:
        self.events = events
        super().__init__(
            events, media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def stream_response(self, send: Send) -> None:
        try:
            await super().stream_response(send)
        finally:
            with CancelScope(shield=True):
                await self.events.aclose()


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("")
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    return await request.app.state.chat.answer(payload)


@router.post("/stream")
async def stream_chat(payload: ChatRequest, request: Request) -> StreamingResponse:
    async def events() -> AsyncGenerator[str, None]:
        try:
            async with aclosing(request.app.state.chat.stream(payload)) as stream:
                async for event, value in stream:
                    data = value.model_dump(mode="json") if isinstance(value, ChatResponse) else {"text": value}
                    yield sse(event, data)
        except AppError as error:
            logger.warning("chat_stream_failed", extra={"error_type": type(error).__name__})
            yield sse("error", {"status": error.status, "detail": error.detail})
        except Exception as error:
            logger.error("chat_stream_failed", extra={"error_type": type(error).__name__})
            yield sse("error", {"status": 500, "detail": "Answer generation failed. Try again."})

    return EventStreamResponse(events())


@router.delete("/sessions/{session_id}", status_code=204)
async def clear_conversation(session_id: UUID, request: Request) -> Response:
    request.app.state.chat.clear(session_id)
    return Response(status_code=204)
