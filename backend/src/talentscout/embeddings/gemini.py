"""Gemini embedding provider used to convert document text into vectors."""

from google import genai
from google.genai import types

from talentscout.config.settings import get_settings
from talentscout.embeddings.service import EmbeddingService


class GeminiEmbeddingService(EmbeddingService):
    """Generate text embeddings using Google's Gemini embedding API."""

    def __init__(self) -> None:
        # Read configuration once when the provider is created.
        settings = get_settings()

        if not settings.gemini_api_key:
            raise ValueError("TALENTSCOUT_GEMINI_API_KEY is not configured")

        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate a single embedding vector for the supplied text."""

        if not text.strip():
            raise ValueError("Cannot generate an embedding for empty text")

        # Gemini's SDK call is synchronous, so run it outside the async event loop.
        import asyncio

        response = await asyncio.to_thread(
            self.client.models.embed_content,
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
            ),
        )

        if not response.embeddings:
            raise RuntimeError("Gemini returned no embedding")

        embedding = response.embeddings[0].values

        if embedding is None or len(embedding) != self.dimensions:
            raise RuntimeError(
                f"Expected {self.dimensions}-dimensional embedding, "
                f"got {len(embedding) if embedding else 0}"
            )

        return list(embedding)
