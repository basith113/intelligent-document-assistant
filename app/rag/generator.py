"""Use the OpenAI Responses API for query rewriting and grounded generation."""

import json
from collections.abc import AsyncIterator

from anyio import CancelScope
from openai import AsyncOpenAI

from app.core.errors import AppError
from app.core.openai_client import provider_errors, require_client
from app.rag.prompts import ANSWER_PROMPT, REWRITE_PROMPT
from app.schemas.chat import Message, Source


class Generator:
    def __init__(self, client: AsyncOpenAI | None, model: str, temperature: float) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    async def rewrite(self, question: str, history: list[Message]) -> str:
        if not history:
            return question
        payload = json.dumps({
            "question": question,
            "history": [message.model_dump() for message in history],
        }, ensure_ascii=False)
        client = require_client(self.client)
        with provider_errors():
            response = await client.responses.create(
                model=self.model,
                instructions=REWRITE_PROMPT,
                input=payload,
                max_output_tokens=512,
                temperature=0,
                store=False,
            )
        if response.status != "completed":
            return question
        rewritten = response.output_text.strip()
        return rewritten if 0 < len(rewritten) <= 2000 else question

    def _answer_input(self, question: str, sources: list[Source]) -> str:
        return json.dumps({
            "question": question,
            "excerpts": [
                {"id": source.citation, "document": source.filename, "page": source.page, "text": source.excerpt}
                for source in sources
            ],
        }, ensure_ascii=False)

    async def answer(self, question: str, sources: list[Source]) -> str:
        client = require_client(self.client)
        with provider_errors():
            response = await client.responses.create(
                model=self.model,
                instructions=ANSWER_PROMPT,
                input=self._answer_input(question, sources),
                max_output_tokens=800,
                temperature=self.temperature,
                store=False,
            )
        if response.status != "completed":
            raise AppError(502, "OpenAI returned an incomplete answer. Try again.")
        return response.output_text

    async def stream(self, question: str, sources: list[Source]) -> AsyncIterator[str]:
        client = require_client(self.client)
        with provider_errors():
            stream = await client.responses.create(
                model=self.model,
                instructions=ANSWER_PROMPT,
                input=self._answer_input(question, sources),
                max_output_tokens=800,
                temperature=self.temperature,
                store=False,
                stream=True,
            )
            try:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield event.delta
                    elif event.type == "response.completed":
                        return
                    elif event.type in {"error", "response.failed", "response.incomplete"}:
                        raise AppError(502, "OpenAI interrupted answer generation. Try again.")
            finally:
                with CancelScope(shield=True):
                    await stream.close()
        raise AppError(502, "The OpenAI stream ended before the answer was complete.")
