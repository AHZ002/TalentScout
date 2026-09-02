"""End-to-end tests for embedding storage and retrieval."""

from uuid import uuid4

import pytest

from talentscout.db.models.document import Document
from talentscout.db.models.document_chunk import DocumentChunk
from talentscout.db.models.job import Job
from talentscout.db.session import SessionFactory
from talentscout.embeddings.jina import JinaEmbeddingService
from talentscout.jobs.repositories.document_chunk import DocumentChunkRepository


@pytest.mark.integration
@pytest.mark.asyncio
async def test_jina_embedding_can_be_stored_and_retrieved() -> None:
    """Verify Jina embeddings work with PostgreSQL pgvector retrieval."""
    embedding_service = JinaEmbeddingService()

    async with SessionFactory() as session:
        # Create the parent job required by the document foreign key.
        job = Job(
            id=uuid4(),
            title="Embedding Integration Test",
            description="Temporary job used to test embedding storage and retrieval.",
        )
        session.add(job)
        await session.flush()

        # Create a document belonging to the test job.
        document = Document(
            id=uuid4(),
            job_id=job.id,
            filename="embedding-test.txt",
            content_type="text/plain",
            storage_path="test/embedding-test.txt",
        )
        session.add(document)
        await session.flush()

        text = "Python is used for backend development and machine learning."

        # Generate a real embedding through Jina.
        embedding = await embedding_service.embed(text)

        assert len(embedding) == 1024

        # Store the real Jina vector in PostgreSQL/pgvector.
        chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            text=text,
            embedding=embedding,
        )
        session.add(chunk)
        await session.flush()

        # Verify pgvector can retrieve the chunk using cosine similarity.
        repository = DocumentChunkRepository(session)

        results = await repository.search(
            job_id=job.id,
            embedding=embedding,
            limit=1,
        )

        assert len(results) == 1
        assert results[0].id == chunk.id

        # Keep the integration test database clean.
        await session.rollback()