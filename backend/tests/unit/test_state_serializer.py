"""Test InterviewState serialization and validation."""

from uuid import uuid4

import pytest
from pydantic import ValidationError

from talentscout.agents.state import InterviewState
from talentscout.agents.state_serializer import (
    deserialize_interview_state,
    serialize_interview_state,
)


def test_interview_state_round_trip() -> None:
    """Verify state can be serialized and restored without data loss."""
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


def test_invalid_persisted_state_is_rejected() -> None:
    """Verify invalid persisted data fails validation."""
    with pytest.raises(ValidationError):
        deserialize_interview_state(
            {
                "job_id": "not-a-valid-uuid",
                "job_description": "Build Python backend services.",
            }
        )
