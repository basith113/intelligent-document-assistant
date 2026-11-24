"""Validation and response models shared by chat endpoints."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, StringConstraints


class ChatRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    session_id: UUID | None = None


class Source(BaseModel):
    citation: str
    document_id: str
    filename: str
    page: int | None
    chunk_id: str
    distance: float
    excerpt: str


class ChatResponse(BaseModel):
    session_id: UUID
    answer: str
    sources: list[Source]


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str
