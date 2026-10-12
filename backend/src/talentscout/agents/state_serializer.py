"""Serialize and validate InterviewState for durable persistence."""

from typing import cast

from pydantic import TypeAdapter

from talentscout.agents.state import InterviewState

_INTERVIEW_STATE_ADAPTER = TypeAdapter(InterviewState)


def serialize_interview_state(state: InterviewState) -> dict[str, object]:
    """Convert an InterviewState into JSON-compatible data."""
    serialized = _INTERVIEW_STATE_ADAPTER.dump_python(
        state,
        mode="json",
    )
    return cast(dict[str, object], serialized)


def deserialize_interview_state(payload: object) -> InterviewState:
    """Validate and reconstruct InterviewState from persisted data."""
    return _INTERVIEW_STATE_ADAPTER.validate_python(payload)
