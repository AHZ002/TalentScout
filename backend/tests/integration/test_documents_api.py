import pytest
from httpx import ASGITransport, AsyncClient

from talentscout.api.main import app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_get_job_document() -> None:
    """Verify that a document can be uploaded and retrieved for a job."""
    job_payload = {
        "title": "Machine Learning Engineer",
        "description": "Build and evaluate machine learning systems.",
        "company_context": ("Healthcare company building patient risk prediction systems."),
    }

    transport = ASGITransport(app=app)

    async with AsyncClient(
        transport=transport,
        base_url="http://test",
    ) as client:
        job_response = await client.post("/jobs", json=job_payload)

        assert job_response.status_code == 201

        job = job_response.json()
        job_id = job["id"]

        files = {
            "file": (
                "clinical_ai_guidelines.txt",
                b"Healthcare AI system guidelines.",
                "text/plain",
            )
        }

        document_response = await client.post(
            f"/jobs/{job_id}/documents",
            files=files,
        )

        assert document_response.status_code == 201

        document = document_response.json()

        assert document["job_id"] == job_id
        assert document["filename"] == "clinical_ai_guidelines.txt"
        assert document["content_type"] == "text/plain"
        assert document["status"] == "completed"

        get_response = await client.get(f"/jobs/{job_id}/documents/{document['id']}")

        assert get_response.status_code == 200
        assert get_response.json() == document
