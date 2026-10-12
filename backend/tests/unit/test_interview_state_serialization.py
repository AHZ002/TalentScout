"""Unit tests for durable InterviewState serialization and validation."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from talentscout.agents.state import InterviewState
from talentscout.agents.state_serializer import (
    deserialize_interview_state,
    serialize_interview_state,
)


def test_interview_state_round_trips_through_json_data() -> None:
    """Verify UUIDs and state values survive persistence serialization."""
    job_id = uuid4()

    state: InterviewState = {
        "job_id": job_id,
        "job_description": "Build Python backend services.",
        "interview_history": [],
        "retrieved_context": [],
    }

    serialized = serialize_interview_state(state)
    restored = deserialize_interview_state(serialized)

    assert serialized["job_id"] == str(job_id)
    assert restored == state


def test_interview_state_deserialization_rejects_invalid_data() -> None:
    """Verify corrupted persisted state fails validation."""
    with pytest.raises(ValidationError):
        deserialize_interview_state(
            {
                "job_id": "not-a-uuid",
                "job_description": "Build Python backend services.",
            }
        )
