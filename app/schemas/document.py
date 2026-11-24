"""Document metadata sent over the REST API."""

from datetime import datetime

from pydantic import BaseModel


class Document(BaseModel):
    document_id: str
    filename: str
    size_bytes: int
    chunk_count: int
    indexed_at: datetime
