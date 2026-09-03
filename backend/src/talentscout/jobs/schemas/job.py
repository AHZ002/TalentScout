from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from talentscout.db.models.job import JobStatus


class JobCreate(BaseModel):
    """Data required to create a new hiring job."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)


class JobResponse(BaseModel):
    """Data returned by the API for a hiring job."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    description: str
    status: JobStatus
