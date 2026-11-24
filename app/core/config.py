"""Environment settings for the API and RAG pipeline."""

from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, StringConstraints, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    chroma_path: Path = Path("./storage/chroma_db")
    collection_name: Annotated[
        str, StringConstraints(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,61}[a-zA-Z0-9]$")
    ] = "knowledge_base"
    chunk_size: int = Field(500, ge=16, le=2000)
    chunk_overlap: int = Field(100, ge=0)
    top_k: int = Field(5, ge=1, le=20)
    max_distance: float = Field(0.8, ge=0.0, le=2.0)
    max_history_messages: int = Field(10, ge=2, le=40)
    max_sessions: int = Field(500, ge=1)
    max_upload_size_mb: int = Field(20, ge=1, le=100)
    temperature: float = Field(0.2, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return self

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024
