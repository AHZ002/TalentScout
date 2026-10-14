"""Manage interview-session lifecycle and validated durable InterviewState persistence."""

from uuid import UUID

from talentscout.agents.state import InterviewState
from talentscout.agents.state_serializer import (
    deserialize_interview_state,
    serialize_interview_state,
)
from talentscout.db.models.interview_session import (
    InterviewSession,
    InterviewSessionStatus,
)
from talentscout.jobs.repositories.interview_session import InterviewSessionRepository
from talentscout.jobs.repositories.job import JobRepository


class JobNotFoundError(LookupError):
    """Raised when an interview is requested for a nonexistent job."""


class InterviewSessionNotFoundError(LookupError):
    """Raised when an interview session does not exist."""


class InterviewSessionAlreadyCompletedError(ValueError):
    """Raised when a caller attempts to resume or modify a completed interview."""


class InvalidInterviewStateError(ValueError):
    """Raised when interview state conflicts with its session or lifecycle."""


class InterviewSessionConflictError(RuntimeError):
    """Raised when another request changes the session during processing."""


class InterviewSessionService:
    """Coordinate job validation, state validation, and session persistence."""

    def __init__(
        self,
        session_repository: InterviewSessionRepository,
        job_repository: JobRepository,
    ) -> None:
        """Initialize the service with repositories sharing the request session."""
        self.session_repository = session_repository
        self.job_repository = job_repository

    async def create_session(self, job_id: UUID) -> InterviewSession:
        """Create an in-progress interview initialized from the selected job."""
        job = await self.job_repository.get_by_id(job_id)
        if job is None:
            raise JobNotFoundError(f"Job '{job_id}' was not found.")

        initial_state: InterviewState = {
            "job_id": job.id,
            "job_description": job.description,
            "interview_history": [],
            "retrieved_context": [],
        }
        validated_state = self._validate_state(
            initial_state,
            expected_job_id=job.id,
        )

        interview_session = InterviewSession(
            job_id=job.id,
            status=InterviewSessionStatus.IN_PROGRESS,
            state=serialize_interview_state(validated_state),
        )
        return await self.session_repository.create(interview_session)

    async def get_session(self, session_id: UUID) -> InterviewSession | None:
        """Return a session after validating its persisted state and lifecycle."""
        interview_session = await self.session_repository.get_by_id(session_id)
        if interview_session is None:
            return None

        state = self._read_session_state(interview_session)
        self._validate_lifecycle(interview_session.status, state)
        return interview_session

    async def resume_session(
        self,
        session_id: UUID,
    ) -> tuple[InterviewSession, InterviewState]:
        """Load validated state for an in-progress interview graph invocation."""
        interview_session = await self.get_session(session_id)
        if interview_session is None:
            raise InterviewSessionNotFoundError(f"Interview session '{session_id}' was not found.")

        if interview_session.status == InterviewSessionStatus.COMPLETED:
            raise InterviewSessionAlreadyCompletedError(
                f"Interview session '{session_id}' is already completed."
            )

        return interview_session, self._read_session_state(interview_session)

    async def save_state(
        self,
        session_id: UUID,
        state: InterviewState,
        *,
        expected_state: dict[str, object],
    ) -> InterviewSession:
        """Validate and persist state only if the session has not changed."""
        interview_session = await self.get_session(session_id)

        if interview_session is None:
            raise InterviewSessionNotFoundError(f"Interview session '{session_id}' was not found.")

        if interview_session.status == InterviewSessionStatus.COMPLETED:
            raise InterviewSessionAlreadyCompletedError(
                f"Interview session '{session_id}' is already completed."
            )

        previous_state = self._read_session_state(interview_session)
        validated_state = self._validate_state(
            state,
            expected_job_id=interview_session.job_id,
        )

        # The job description is the source-of-truth snapshot captured at
        # session creation and must remain unchanged throughout the interview.
        if validated_state["job_description"] != previous_state["job_description"]:
            raise InvalidInterviewStateError(
                "The job description cannot change during an interview session."
            )

        new_status = (
            InterviewSessionStatus.COMPLETED
            if validated_state.get("interview_complete", False)
            else InterviewSessionStatus.IN_PROGRESS
        )

        self._validate_lifecycle(new_status, validated_state)

        # The repository compares expected_state and updates the row in one
        # SQL statement. Never overwrite the ORM object before that check.
        saved_session = await self.session_repository.update_if_unchanged(
            interview_session,
            expected_state=expected_state,
            new_state=serialize_interview_state(validated_state),
            new_status=new_status,
        )

        if saved_session is None:
            raise InterviewSessionConflictError(
                "The interview session changed while the answer was being "
                "processed. Retrieve the latest session state before retrying."
            )

        return saved_session

    @staticmethod
    def _validate_state(
        state: InterviewState,
        expected_job_id: UUID,
    ) -> InterviewState:
        """Round-trip state through Pydantic and enforce required session identity."""
        validated_state = deserialize_interview_state(serialize_interview_state(state))
        state_job_id = validated_state.get("job_id")
        job_description = validated_state.get("job_description")

        if state_job_id is None or job_description is None or not job_description.strip():
            raise InvalidInterviewStateError(
                "Interview state must include a job ID and a non-empty job description."
            )
        if state_job_id != expected_job_id:
            raise InvalidInterviewStateError(
                "Interview state job ID does not match the session's job ID."
            )

        return validated_state

    @classmethod
    def _read_session_state(
        cls,
        interview_session: InterviewSession,
    ) -> InterviewState:
        """Deserialize stored JSON and verify it belongs to this database session."""
        try:
            state = deserialize_interview_state(interview_session.state)
        except (TypeError, ValueError) as exc:
            raise InvalidInterviewStateError("Persisted interview state is invalid.") from exc

        return cls._validate_state(state, expected_job_id=interview_session.job_id)

    @staticmethod
    def _validate_lifecycle(
        status_value: InterviewSessionStatus | str,
        state: InterviewState,
    ) -> None:
        """Ensure completion status agrees with the final state and report."""
        interview_complete = state.get("interview_complete", False)
        has_report = "assessment_report" in state

        if interview_complete != has_report:
            raise InvalidInterviewStateError(
                "Interview completion and assessment-report state are inconsistent."
            )

        state_is_completed = interview_complete and has_report
        session_is_completed = status_value == InterviewSessionStatus.COMPLETED
        if state_is_completed != session_is_completed:
            raise InvalidInterviewStateError(
                "Interview session status does not match its persisted state."
            )
