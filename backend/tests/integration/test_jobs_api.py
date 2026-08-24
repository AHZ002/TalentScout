import pytest
from httpx import ASGITransport, AsyncClient

from talentscout.api.main import app


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_and_get_job() -> None:
    """Verify that a job can be created and retrieved through the API."""
    payload = {
        "title": "Machine Learning Engineer",
        "description": "Build and evaluate machine learning systems.",
        "company_context": ("Healthcare company building patient risk prediction systems."),
    }

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
        assert created_job["company_context"] == payload["company_context"]
        assert created_job["status"] == "draft"

        job_id = created_job["id"]

        get_response = await client.get(f"/jobs/{job_id}")

        assert get_response.status_code == 200
        assert get_response.json() == created_job
