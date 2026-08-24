from uuid import UUID

from talentscout.db.models.job import Job
from talentscout.jobs.repositories.job import JobRepository
from talentscout.jobs.schemas.job import JobCreate


class JobService:
    """Contains business logic for hiring jobs."""

    def __init__(self, repository: JobRepository) -> None:
        self.repository = repository

    async def create_job(self, data: JobCreate) -> Job:
        """Create and persist a new hiring job."""
        job = Job(
            title=data.title,
            description=data.description,
            company_context=data.company_context,
        )

        return await self.repository.create(job)

    async def get_job(self, job_id: UUID) -> Job | None:
        """Retrieve a hiring job by ID."""
        return await self.repository.get_by_id(job_id)
