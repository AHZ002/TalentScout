from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from talentscout.db.models.document import DocumentStatus


class DocumentCreate(BaseModel):
    """Data required to register a company document."""

    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=100)
    storage_path: str = Field(min_length=1, max_length=500)


class DocumentResponse(BaseModel):
    """Data returned by the API for an Additional Interview Guidance document."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    job_id: UUID
    filename: str
    content_type: str
    storage_path: str
    status: DocumentStatus
