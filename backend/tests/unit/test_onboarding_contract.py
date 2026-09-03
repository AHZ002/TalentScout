"""Tests for the hiring-company onboarding contract."""

import pytest
from pydantic import ValidationError

from talentscout.api.main import create_app
from talentscout.jobs.schemas.job import JobCreate


def test_job_creation_accepts_only_the_required_job_description_inputs() -> None:
    """Ensure retired company context cannot be submitted for a new job."""
    job = JobCreate(
        title="Backend Engineer",
        description="Build reliable Python services.",
    )

    assert job.title == "Backend Engineer"

    with pytest.raises(ValidationError, match="company_context"):
        JobCreate.model_validate(
            {
                "title": "Backend Engineer",
                "description": "Build reliable Python services.",
                "company_context": "This field must not be accepted.",
            }
        )


def test_api_exposes_only_the_additional_interview_guidance_route() -> None:
    """Ensure onboarding terminology is reflected in the public API contract."""
    paths = create_app().openapi()["paths"]

    assert "/jobs/{job_id}/additional-interview-guidance" in paths
    assert "/jobs/{job_id}/documents" not in paths
