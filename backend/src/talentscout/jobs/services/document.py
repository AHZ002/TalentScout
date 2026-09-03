"""Business logic for managing Additional Interview Guidance documents.

Coordinates document storage and database operations.
"""

from uuid import UUID

import httpx

from talentscout.db.models.document import Document, DocumentStatus
from talentscout.db.models.document_chunk import DocumentChunk
from talentscout.documents.chunker import DocumentChunker
from talentscout.documents.processor import DocumentProcessor
from talentscout.embeddings.service import EmbeddingService
from talentscout.jobs.repositories.document import DocumentRepository
from talentscout.jobs.repositories.document_chunk import DocumentChunkRepository
from talentscout.storage.base import StorageService


class DocumentService:
    """Handles storing and processing Additional Interview Guidance."""

    def __init__(
        self,
        repository: DocumentRepository,
        storage: StorageService,
        processor: DocumentProcessor,
        chunker: DocumentChunker,
        chunk_repository: DocumentChunkRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.processor = processor
        self.chunker = chunker
        self.chunk_repository = chunk_repository
        self.embedding_service = embedding_service

    async def create_guidance_document(
        self,
        job_id: UUID,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> Document:
        """Store an Additional Interview Guidance document and process its text."""
        storage_path = await self.storage.save(
            content=content,
            filename=filename,
            content_type=content_type,
        )

        document = Document(
            job_id=job_id,
            filename=filename,
            content_type=content_type,
            storage_path=storage_path,
            status=DocumentStatus.PROCESSING,
        )

        try:
            extracted_text = await self.processor.extract_text(
                content=content,
                content_type=content_type,
            )

            # Split the extracted text into retrieval-ready pieces.
            chunks = self.chunker.chunk(extracted_text)

            # Persist the document while processing is still in progress.
            # This allows us to retain a FAILED document record if embedding fails.
            document = await self.repository.create(document)

            # Generate vectors for all chunks before marking processing as complete.
            embeddings = await self.embedding_service.embed_many(
                [chunk.text for chunk in chunks]
            )

            # Store each chunk together with its semantic embedding.
            document_chunks = [
                DocumentChunk(
                    document_id=document.id,
                    chunk_index=chunk.index,
                    text=chunk.text,
                    embedding=embedding,
                )
                for chunk, embedding in zip(chunks, embeddings, strict=True)
            ]

            if document_chunks:
                await self.chunk_repository.create_many(document_chunks)

            # Only mark the document complete after all chunks and embeddings succeed.
            document.status = DocumentStatus.COMPLETED

            return document

        except (
            UnicodeDecodeError,
            ValueError,
            RuntimeError,
            httpx.HTTPError,
        ):
            # Mark the document as failed when an expected external-processing
            # or document-processing error occurs.
            document.status = DocumentStatus.FAILED
            return await self.repository.create(document)

    async def reembed_missing_chunks(self) -> int:
        """Generate embeddings for existing chunks that do not have one."""
        chunks = await self.chunk_repository.list_without_embeddings()

        if not chunks:
            return 0

        # Generate embeddings through the provider-independent abstraction.
        embeddings = await self.embedding_service.embed_many(
            [chunk.text for chunk in chunks]
        )

        # Attach each generated vector to its existing chunk.
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk.embedding = embedding

        return len(chunks)    

    async def get_guidance_document(self, document_id: UUID) -> Document | None:
        """Retrieve an Additional Interview Guidance document by ID."""
        return await self.repository.get_by_id(document_id)

    async def list_guidance_documents(self, job_id: UUID) -> list[Document]:
        """Retrieve all Additional Interview Guidance for a hiring job."""
        return await self.repository.list_by_job(job_id)
