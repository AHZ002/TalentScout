from pathlib import Path
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from talentscout.api.main import app
from talentscout.db.models.job import Job
from talentscout.db.session import SessionFactory


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_get_job_document() -> None:
    """Verify that a document can be uploaded and retrieved for a job."""
    job_payload = {
        "title": "Machine Learning Engineer",
        "description": "Build and evaluate machine learning systems.",
    }

    job_id: UUID | None = None
    storage_path: Path | None = None

    try:
        transport = ASGITransport(app=app)

        async with AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            job_response = await client.post("/jobs", json=job_payload)

            assert job_response.status_code == 201

            job = job_response.json()
            job_id = UUID(job["id"])

            files = {
                "file": (
                    "clinical_ai_guidelines.txt",
                    b"Healthcare AI system guidelines.",
                    "text/plain",
                )
            }

            document_response = await client.post(
                f"/jobs/{job_id}/additional-interview-guidance",
                files=files,
            )

            assert document_response.status_code == 201

            document = document_response.json()
            storage_path = Path(document["storage_path"])

            assert document["job_id"] == str(job_id)
            assert document["filename"] == "clinical_ai_guidelines.txt"
            assert document["content_type"] == "text/plain"
            assert document["status"] == "completed"

            get_response = await client.get(
                f"/jobs/{job_id}/additional-interview-guidance/{document['id']}"
            )

            assert get_response.status_code == 200
            assert get_response.json() == document
    finally:
        if storage_path is not None:
            storage_path.unlink(missing_ok=True)

        if job_id is not None:
            async with SessionFactory() as session:
                await session.execute(delete(Job).where(Job.id == job_id))
                await session.commit()
