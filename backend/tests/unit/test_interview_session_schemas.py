"""Unit tests for interview-session API request validation."""

import pytest
from pydantic import ValidationError

from talentscout.jobs.schemas.interview_session import InterviewAnswerSubmit


def test_candidate_answer_is_trimmed() -> None:
    """Normalize surrounding whitespace before processing an answer."""
    request = InterviewAnswerSubmit(
        candidate_answer="  I would use FastAPI.  ",
    )

    assert request.candidate_answer == "I would use FastAPI."


def test_whitespace_only_candidate_answer_is_rejected() -> None:
    """Reject answers that contain no meaningful content."""
    with pytest.raises(ValidationError):
        InterviewAnswerSubmit(candidate_answer="   ")


def test_empty_candidate_answer_is_rejected() -> None:
    """Reject an empty answer."""
    with pytest.raises(ValidationError):
        InterviewAnswerSubmit(candidate_answer="")


def test_unexpected_request_fields_are_rejected() -> None:
    """Prevent undocumented fields from silently entering the API."""
    with pytest.raises(ValidationError):
        InterviewAnswerSubmit(
            candidate_answer="I would use FastAPI.",
            job_id="unexpected",
        )
