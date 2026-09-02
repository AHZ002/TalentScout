"""Jina embedding provider used to convert document text into vectors."""

import httpx

from talentscout.config.settings import get_settings
from talentscout.embeddings.service import EmbeddingService


class JinaEmbeddingService(EmbeddingService):
    """Generate text embeddings using Jina AI's embeddings API."""

    API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self) -> None:
        # Load the API key and embedding configuration when the provider starts.
        settings = get_settings()

        if not settings.jina_api_key:
            raise ValueError("TALENTSCOUT_JINA_API_KEY is not configured")

        self.api_key = settings.jina_api_key
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a single piece of text."""

        if not text.strip():
            raise ValueError("Cannot generate an embedding for empty text")

        # Jina exposes a standard HTTP API, so use an async HTTP client
        # without blocking the application's event loop.
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "input": [text],
                },
            )

        response.raise_for_status()

        data = response.json()
        embeddings = data.get("data", [])

        if not embeddings:
            raise RuntimeError("Jina returned no embedding")

        embedding = embeddings[0].get("embedding")

        if not isinstance(embedding, list) or len(embedding) != self.dimensions:
            actual_dimensions = len(embedding) if isinstance(embedding, list) else 0
            raise RuntimeError(
                f"Expected {self.dimensions}-dimensional embedding, got {actual_dimensions}"
            )

        return [float(value) for value in embedding]
