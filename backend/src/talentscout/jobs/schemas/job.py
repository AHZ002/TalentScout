from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from talentscout.db.models.job import JobStatus


class JobCreate(BaseModel):
    """Data required to create a new hiring job."""

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)
    company_context: str | None = None


class JobResponse(BaseModel):
    """Data returned by the API for a hiring job."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    description: str
    company_context: str | None
    status: JobStatus
