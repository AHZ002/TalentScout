"""Semantic retrieval of company-document context for a hiring job."""

from uuid import UUID

from talentscout.db.models.document_chunk import DocumentChunk
from talentscout.embeddings.service import EmbeddingService
from talentscout.jobs.repositories.document_chunk import DocumentChunkRepository


class DocumentRetriever:
    """Find the most relevant company-document chunks for a query."""

    def __init__(
        self,
        repository: DocumentChunkRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        # The repository handles PostgreSQL/pgvector search.
        self.repository = repository

        # The embedding service converts the user's query into a vector.
        self.embedding_service = embedding_service

    async def retrieve(
        self,
        job_id: UUID,
        query: str,
        limit: int = 5,
    ) -> list[DocumentChunk]:
        """Return the most relevant document chunks for a job."""

        if not query.strip():
            raise ValueError("Retrieval query cannot be empty")

        # Use the same embedding model that was used for document chunks.
        query_embedding = await self.embedding_service.embed(query)

        # Let the repository perform the actual vector similarity search.
        return await self.repository.search(
            job_id=job_id,
            embedding=query_embedding,
            limit=limit,
        )
