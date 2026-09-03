from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from talentscout.db.models.document import Document
from talentscout.db.session import get_session
from talentscout.documents.chunker import DocumentChunker
from talentscout.documents.processors.basic import BasicDocumentProcessor
from talentscout.embeddings.jina import JinaEmbeddingService
from talentscout.jobs.repositories.document import DocumentRepository
from talentscout.jobs.repositories.document_chunk import DocumentChunkRepository
from talentscout.jobs.schemas.document import DocumentResponse
from talentscout.jobs.services.document import DocumentService
from talentscout.storage.local import LocalStorageService

router = APIRouter(
    prefix="/jobs/{job_id}/additional-interview-guidance",
    tags=["additional-interview-guidance"],
)


def get_document_service(
    session: AsyncSession = Depends(get_session),
) -> DocumentService:
    """Build the guidance service with database and file storage access."""
    repository = DocumentRepository(session)
    storage = LocalStorageService(Path("storage/documents"))

    processor = BasicDocumentProcessor()

    chunker = DocumentChunker()
    chunk_repository = DocumentChunkRepository(session)

    # Provides semantic embeddings for document chunks.
    embedding_service = JinaEmbeddingService()

    return DocumentService(
        repository=repository,
        storage=storage,
        processor=processor,
        chunker=chunker,
        chunk_repository=chunk_repository,
        embedding_service=embedding_service,
    )


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_guidance_document(
    job_id: UUID,
    file: UploadFile = File(...),
    service: DocumentService = Depends(get_document_service),
) -> Document:
    """Upload optional Additional Interview Guidance for a hiring job."""
    content = await file.read()

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename",
        )

    return await service.create_guidance_document(
        job_id=job_id,
        content=content,
        filename=file.filename,
        content_type=file.content_type or "application/octet-stream",
    )


@router.get(
    "",
    response_model=list[DocumentResponse],
)
async def list_guidance_documents(
    job_id: UUID,
    service: DocumentService = Depends(get_document_service),
) -> list[Document]:
    """List all Additional Interview Guidance for a hiring job."""
    return await service.list_guidance_documents(job_id)


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
)
async def get_guidance_document(
    job_id: UUID,
    document_id: UUID,
    service: DocumentService = Depends(get_document_service),
) -> Document:
    """Retrieve an Additional Interview Guidance document for a hiring job."""
    document = await service.get_guidance_document(document_id)

    if document is None or document.job_id != job_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Additional Interview Guidance document not found",
        )

    return document
