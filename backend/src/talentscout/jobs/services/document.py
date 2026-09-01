"""Business logic for managing company documents.

Coordinates document storage and database operations.
"""

from uuid import UUID

from talentscout.db.models.document import Document, DocumentStatus
from talentscout.db.models.document_chunk import DocumentChunk
from talentscout.documents.chunker import DocumentChunker
from talentscout.documents.processor import DocumentProcessor
from talentscout.embeddings.service import EmbeddingService
from talentscout.jobs.repositories.document import DocumentRepository
from talentscout.jobs.repositories.document_chunk import DocumentChunkRepository
from talentscout.storage.base import StorageService


class DocumentService:
    """Handles storing and processing company documents."""

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

    async def create_document(
        self,
        job_id: UUID,
        content: bytes,
        filename: str,
        content_type: str,
    ) -> Document:
        """Store a document and process its text."""
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

            document.status = DocumentStatus.COMPLETED

            document = await self.repository.create(
                document
            )  # This sends the Document object to the repository.
            # The repository then saves it into the documents database table.

            # Generate vectors for all chunks before storing them.
            embeddings = await self.embedding_service.embed_many([chunk.text for chunk in chunks])

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

            return document

        except (UnicodeDecodeError, ValueError):
            document.status = DocumentStatus.FAILED
            return await self.repository.create(document)

    async def get_document(self, document_id: UUID) -> Document | None:
        """Retrieve a document by ID."""
        return await self.repository.get_by_id(document_id)

    async def list_job_documents(self, job_id: UUID) -> list[Document]:
        """Retrieve all documents belonging to a hiring job."""
        return await self.repository.list_by_job(job_id)
