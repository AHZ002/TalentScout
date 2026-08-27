"""Business logic for managing company documents.

Coordinates document storage and database operations.
"""

from uuid import UUID

from talentscout.db.models.document import Document, DocumentStatus
from talentscout.documents.processor import DocumentProcessor
from talentscout.jobs.repositories.document import DocumentRepository
from talentscout.storage.base import StorageService


class DocumentService:
    """Handles storing and processing company documents."""

    def __init__(
        self,
        repository: DocumentRepository,
        storage: StorageService,
        processor: DocumentProcessor,
    ) -> None:
        self.repository = repository
        self.storage = storage
        self.processor = processor

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
            await self.processor.extract_text(
                content=content,
                content_type=content_type,
            )
            document.status = DocumentStatus.COMPLETED
        except (UnicodeDecodeError, ValueError):
            document.status = DocumentStatus.FAILED

        return await self.repository.create(
            document
        )  # This sends the Document object to the repository.
        # The repository then saves it into the documents database table.

    async def get_document(self, document_id: UUID) -> Document | None:
        """Retrieve a document by ID."""
        return await self.repository.get_by_id(document_id)

    async def list_job_documents(self, job_id: UUID) -> list[Document]:
        """Retrieve all documents belonging to a hiring job."""
        return await self.repository.list_by_job(job_id)
