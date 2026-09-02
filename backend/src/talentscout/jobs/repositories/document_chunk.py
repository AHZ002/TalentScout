"""Save multiple chunks to PostgreSQL.
Search those chunks for the ones most similar to a user's question."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from talentscout.db.models.document_chunk import DocumentChunk


class DocumentChunkRepository:
    """Persists and retrieves document chunks."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_many(
        self,
        chunks: list[DocumentChunk],
    ) -> list[DocumentChunk]:
        """Persist multiple document chunks."""
        self.session.add_all(chunks)
        await self.session.flush()
        return chunks

    async def list_without_embeddings(self) -> list[DocumentChunk]:
        """Return document chunks that do not have an embedding yet."""
        statement = (
            select(DocumentChunk)
            .where(DocumentChunk.embedding.is_(None))
            .order_by(DocumentChunk.created_at)
        )

        result = await self.session.scalars(statement)
        return list(result)

    async def search(
        self,
        job_id: UUID,
        embedding: list[float],
        limit: int = 5,
    ) -> list[DocumentChunk]:
        """Retrieve the most similar chunks belonging to a job."""
        statement = (
            select(DocumentChunk)
            .join(DocumentChunk.document)
            .where(DocumentChunk.document.has(job_id=job_id))
            .order_by(DocumentChunk.embedding.cosine_distance(embedding))
            .limit(limit)
        )

        result = await self.session.scalars(statement)
        return list(result)