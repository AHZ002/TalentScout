from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from talentscout.db.models.document import Document


class DocumentRepository:
    """Handles database operations for company documents."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, document: Document) -> Document:
        """Save a new document record to the database."""
        self.session.add(document)
        await self.session.flush()
        await self.session.refresh(document)
        return document

    async def get_by_id(self, document_id: UUID) -> Document | None:
        """Find a document by its ID."""
        result = await self.session.execute(select(Document).where(Document.id == document_id))
        return result.scalar_one_or_none()

    async def list_by_job(self, job_id: UUID) -> list[Document]:
        """Return all documents associated with a job."""
        result = await self.session.execute(
            select(Document).where(Document.job_id == job_id).order_by(Document.created_at)
        )
        return list(result.scalars().all())
