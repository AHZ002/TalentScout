from abc import ABC, abstractmethod


class EmbeddingService(ABC):
    """Converts text into vector embeddings."""

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate an embedding for a piece of text."""
        raise NotImplementedError

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.embed(text) for text in texts]
