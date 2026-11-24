"""Generate embeddings in bounded batches using the async OpenAI SDK."""

from openai import AsyncOpenAI

from app.core.errors import AppError
from app.core.openai_client import provider_errors, require_client


class Embedder:
    def __init__(self, client: AsyncOpenAI | None, model: str) -> None:
        self.client = client
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return vectors in input order. The SDK retries transient API failures."""
        if not texts:
            return []
        client = require_client(self.client)
        vectors = []
        batch_size = 64
        with provider_errors():
            for start in range(0, len(texts), batch_size):
                batch = texts[start:start + batch_size]
                result = await client.embeddings.create(model=self.model, input=batch)
                items = sorted(result.data, key=lambda item: item.index)
                if [item.index for item in items] != list(range(len(batch))):
                    raise AppError(502, "OpenAI returned an incomplete embedding batch.")
                vectors.extend(item.embedding for item in items)
        return vectors
