"""One-time maintenance command for generating missing document embeddings."""

import asyncio
import sys

from talentscout.db.session import SessionFactory
from talentscout.embeddings.jina import JinaEmbeddingService
from talentscout.jobs.repositories.document_chunk import DocumentChunkRepository


async def reembed_missing_chunks() -> None:
    """Generate embeddings for all existing chunks that are missing them."""
    async with SessionFactory() as session:
        chunk_repository = DocumentChunkRepository(session)

        chunks = await chunk_repository.list_without_embeddings()

        if not chunks:
            print("No document chunks are missing embeddings.")
            return

        print(f"Found {len(chunks)} chunks without embeddings.")

        embedding_service = JinaEmbeddingService()

        embeddings = await embedding_service.embed_many(
            [chunk.text for chunk in chunks]
        )

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding

        await session.commit()

        print(f"Successfully embedded {len(chunks)} chunks.")


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(reembed_missing_chunks())