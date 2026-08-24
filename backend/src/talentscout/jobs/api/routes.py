from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from talentscout.db.models.job import Job
from talentscout.db.session import get_session
from talentscout.jobs.repositories.job import JobRepository
from talentscout.jobs.schemas.job import JobCreate, JobResponse
from talentscout.jobs.services.job import JobService

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_job_service(
    session: AsyncSession = Depends(get_session),
) -> JobService:
    """Build the Job service using the current database session."""
    repository = JobRepository(session)
    return JobService(repository)


@router.post(
    "",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_job(
    data: JobCreate,
    service: JobService = Depends(get_job_service),
) -> Job:
    """Create a new hiring job."""
    return await service.create_job(data)


@router.get(
    "/{job_id}",
    response_model=JobResponse,
)
async def get_job(
    job_id: UUID,
    service: JobService = Depends(get_job_service),
) -> Job:
    """Retrieve a hiring job by ID."""
    job = await service.get_job(job_id)

    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        )

    return job
