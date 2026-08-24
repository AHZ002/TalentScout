from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from talentscout.db.models.document import Document
from talentscout.db.session import get_session
from talentscout.jobs.repositories.document import DocumentRepository
from talentscout.jobs.schemas.document import DocumentResponse
from talentscout.jobs.services.document import DocumentService
from talentscout.storage.local import LocalStorageService

router = APIRouter(
    prefix="/jobs/{job_id}/documents",
    tags=["documents"],
)


def get_document_service(
    session: AsyncSession = Depends(get_session),
) -> DocumentService:
    """Build the Document service with database and file storage access."""
    repository = DocumentRepository(session)
    storage = LocalStorageService(Path("storage/documents"))

    return DocumentService(
        repository=repository,
        storage=storage,
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_document(
    job_id: UUID,
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> Document:
    """Upload and register a company document."""
    content = await file.read()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename",
        )

    return await service.create_document(
        job_id=job_id,
        content=content,
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
    )


@router.get(
    "",
    response_model=list[DocumentResponse],
)
async def list_documents(
    job_id: UUID,
    service: DocumentService = Depends(get_document_service),
) -> list[Document]:
    """List all documents belonging to a hiring job."""
    return await service.list_job_documents(job_id)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
async def get_document(
    job_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
) -> Document:
    """Retrieve a document belonging to a hiring job."""
    document = await service.get_document(document_id)

    if document is None or document.job_id != job_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return document
