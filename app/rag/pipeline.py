"""Connect retrieval, query rewriting, answer generation, and citation validation."""

import re

from app.rag.generator import Generator
from app.rag.prompts import INSUFFICIENT
from app.rag.retriever import Retriever
from app.schemas.chat import Message, Source

MARKER = re.compile(r"\[S\d+\]")


class Pipeline:
    def __init__(self, retriever: Retriever, generator: Generator) -> None:
        self.retriever = retriever
        self.generator = generator

    async def prepare(self, question: str, history: list[Message]) -> tuple[str, list[Source]]:
        if not await self.retriever.has_documents():
            return question, []
        query = await self.generator.rewrite(question, history)
        return query, await self.retriever.retrieve(query)

    @staticmethod
    def finalize(answer: str, sources: list[Source]) -> tuple[str, list[Source]]:
        """Return only real source references, in order of first citation."""
        by_marker = {f"[{source.citation}]": source for source in sources}
        markers = list(dict.fromkeys(MARKER.findall(answer)))
        if not markers or any(marker not in by_marker for marker in markers):
            return INSUFFICIENT, []
        return answer.strip(), [by_marker[marker] for marker in markers]
