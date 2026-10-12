"""Pydantic request and response schemas for interview sessions."""

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from talentscout.agents.assessment_report import AssessmentReport
from talentscout.db.models.interview_session import InterviewSessionStatus


class InterviewAnswerSubmit(BaseModel):
    """Validate a candidate's answer submitted to an interview session."""

    model_config = ConfigDict(extra="forbid")

    candidate_answer: str = Field(min_length=1)

    @field_validator("candidate_answer")
    @classmethod
    def normalize_candidate_answer(cls, value: str) -> str:
        """Trim surrounding whitespace and reject blank answers."""
        normalized = value.strip()

        if not normalized:
            raise ValueError("Candidate answer must not be blank.")

        return normalized


class InterviewSessionResponse(BaseModel):
    """Expose the candidate-facing state of an interview session."""

    model_config = ConfigDict(extra="forbid")

    id: UUID
    job_id: UUID
    status: InterviewSessionStatus
    current_question: str | None = None
    interview_complete: bool
    turn_count: int
    assessment_report: AssessmentReport | None = None
