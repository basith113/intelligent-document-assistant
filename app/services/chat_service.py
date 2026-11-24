"""Manage bounded session history and complete or streaming chat turns."""

from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator
from contextlib import aclosing, contextmanager
from typing import Literal
from uuid import UUID, uuid4

from app.core.errors import AppError
from app.rag.pipeline import Pipeline
from app.rag.prompts import INSUFFICIENT
from app.schemas.chat import ChatRequest, ChatResponse, Message, Source


class ChatService:
    def __init__(self, pipeline: Pipeline, history_limit: int, session_limit: int) -> None:
        self.pipeline = pipeline
        self.history_limit = history_limit - history_limit % 2
        self.session_limit = session_limit
        self.histories: OrderedDict[UUID, list[Message]] = OrderedDict()
        self.active: set[UUID] = set()

    @contextmanager
    def turn(self, session_id: UUID) -> Iterator[list[Message]]:
        if session_id in self.active:
            raise AppError(409, "This conversation already has a question in progress.")
        self.active.add(session_id)
        try:
            yield self.histories.get(session_id, [])
        finally:
            self.active.remove(session_id)

    def clear(self, session_id: UUID) -> None:
        with self.turn(session_id):
            self.histories.pop(session_id, None)

    def _complete(
        self, session_id: UUID, request: ChatRequest, history: list[Message],
        answer: str, sources: list[Source],
    ) -> ChatResponse:
        answer, citations = self.pipeline.finalize(answer, sources)
        messages = [
            *history,
            Message(role="user", content=request.question),
            Message(role="assistant", content=answer),
        ]
        self.histories[session_id] = messages[-self.history_limit:]
        self.histories.move_to_end(session_id)
        if len(self.histories) > self.session_limit:
            self.histories.popitem(last=False)
        return ChatResponse(session_id=session_id, answer=answer, sources=citations)

    async def answer(self, request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or uuid4()
        with self.turn(session_id) as history:
            query, sources = await self.pipeline.prepare(request.question, history)
            answer = await self.pipeline.generator.answer(query, sources) if sources else INSUFFICIENT
            return self._complete(session_id, request, history, answer, sources)

    async def stream(
        self, request: ChatRequest
    ) -> AsyncIterator[tuple[Literal["delta", "done"], str | ChatResponse]]:
        session_id = request.session_id or uuid4()
        with self.turn(session_id) as history:
            query, sources = await self.pipeline.prepare(request.question, history)
            parts = []
            if sources:
                async with aclosing(self.pipeline.generator.stream(query, sources)) as stream:
                    async for delta in stream:
                        parts.append(delta)
                        yield "delta", delta
            answer = self._complete(session_id, request, history, "".join(parts), sources)
        yield "done", answer
