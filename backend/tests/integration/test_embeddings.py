"""Integration tests for the Jina embedding provider."""

import pytest

from talentscout.embeddings.jina import JinaEmbeddingService


@pytest.mark.integration
@pytest.mark.asyncio
async def test_jina_generates_1024_dimensional_embedding() -> None:
    """Verify that Jina returns a valid 1024-dimensional embedding."""
    service = JinaEmbeddingService()

    embedding = await service.embed(
        "A technical interview question about Python and machine learning."
    )

    # The configured Jina model must return the dimension used by pgvector.
    assert len(embedding) == 1024

    # Embeddings should contain numeric values suitable for pgvector.
    assert all(isinstance(value, float) for value in embedding)