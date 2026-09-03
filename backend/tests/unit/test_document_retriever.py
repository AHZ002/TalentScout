"""Tests for semantic retrieval of Additional Interview Guidance."""

from typing import cast
from uuid import UUID, uuid4

import pytest

from talentscout.db.models.document_chunk import DocumentChunk
from talentscout.documents.retriever import DocumentRetriever
from talentscout.embeddings.service import EmbeddingService
from talentscout.jobs.repositories.document_chunk import DocumentChunkRepository


class FakeEmbeddingService:
    """Provides deterministic embeddings for the retrieval test."""

    async def embed(self, text: str) -> list[float]:
        """Return a fixed vector for the supplied query."""
        return [1.0, 0.0, 0.0]


class FakeChunkRepository:
    """Records retrieval requests without requiring PostgreSQL."""

    def __init__(self) -> None:
        self.job_id: UUID | None = None
        self.embedding: list[float] | None = None
        self.limit: int | None = None

    async def search(
        self,
        job_id: UUID,
        embedding: list[float],
        limit: int,
    ) -> list[DocumentChunk]:
        """Capture the search parameters and return no chunks."""
        self.job_id = job_id
        self.embedding = embedding
        self.limit = limit
        return []


@pytest.mark.asyncio
async def test_retriever_embeds_query_and_searches_repository() -> None:
    """Verify that retrieval embeds the query and searches the repository."""
    repository = FakeChunkRepository()
    embedding_service = FakeEmbeddingService()
    retriever = DocumentRetriever(
        cast(DocumentChunkRepository, repository),
        cast(EmbeddingService, embedding_service),
    )

    job_id = uuid4()

    results = await retriever.retrieve(
        job_id=job_id,
        query="How does the patient risk system work?",
        limit=3,
    )

    assert results == []
    assert repository.job_id == job_id
    assert repository.embedding == [1.0, 0.0, 0.0]
    assert repository.limit == 3
