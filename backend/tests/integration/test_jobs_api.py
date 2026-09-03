from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from talentscout.api.main import app
from talentscout.db.models.job import Job
from talentscout.db.session import SessionFactory


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_get_job() -> None:
    """Verify that a job can be created and retrieved through the API."""
    payload = {
        "title": "Machine Learning Engineer",
        "description": "Build and evaluate machine learning systems.",
    }

    job_id: UUID | None = None

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            create_response = await client.post("/jobs", json=payload)

            assert create_response.status_code == 201

            created_job = create_response.json()

            assert created_job["title"] == payload["title"]
            assert created_job["description"] == payload["description"]
            assert "company_context" not in created_job
            assert created_job["status"] == "draft"

            job_id = UUID(created_job["id"])

            get_response = await client.get(f"/jobs/{job_id}")

            assert get_response.status_code == 200
            assert get_response.json() == created_job

            rejected_response = await client.post(
                "/jobs",
                json={
                    **payload,
                    "company_context": "Legacy company information is no longer accepted.",
                },
            )

            assert rejected_response.status_code == 422
    finally:
        if job_id is not None:
            async with SessionFactory() as session:
                await session.execute(delete(Job).where(Job.id == job_id))
                await session.commit()
