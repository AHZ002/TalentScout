"""Unit tests for interview-session creation, resumption, and state persistence."""

from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from talentscout.agents.assessment_report import AssessmentReport
from talentscout.agents.state import InterviewState
from talentscout.db.models.interview_session import (
    InterviewSession,
    InterviewSessionStatus,
)
from talentscout.jobs.services.interview_session import (
    InterviewSessionAlreadyCompletedError,
    InterviewSessionConflictError,
    InterviewSessionNotFoundError,
    InterviewSessionService,
    InvalidInterviewStateError,
    JobNotFoundError,
)


def _initial_state(job_id: UUID) -> InterviewState:
    """Build a minimal valid graph state for service tests."""
    return {
        "job_id": job_id,
        "job_description": "Build Python backend services.",
        "interview_history": [],
        "retrieved_context": [],
    }


def _assessment_report() -> AssessmentReport:
    """Build a valid report without invoking an LLM provider."""
    return AssessmentReport(
        overall_summary="The candidate demonstrated strong Python fundamentals.",
        competency_assessments=[
            {
                "name": "Python",
                "performance": "strong",
                "evidence": ["Explained Python data structures accurately."],
                "gaps": [],
            }
        ],
        strengths=["Clear technical reasoning"],
        gaps=[],
        follow_up_areas=[],
        confidence=0.9,
    )


@pytest.fixture
def service_dependencies() -> tuple[InterviewSessionService, AsyncMock, AsyncMock]:
    """Provide mocked repositories so unit tests require no database."""
    from talentscout.jobs.repositories.interview_session import InterviewSessionRepository
    from talentscout.jobs.repositories.job import JobRepository

    session_repository = AsyncMock(spec=InterviewSessionRepository)
    job_repository = AsyncMock(spec=JobRepository)
    session_repository.create.side_effect = lambda value: value

    async def persist_if_unchanged(
        interview_session: InterviewSession,
        *,
        expected_state: dict[str, object],
        new_state: dict[str, object],
        new_status: InterviewSessionStatus,
    ) -> InterviewSession:
        """Simulate a successful conditional persistence operation."""
        interview_session.state = new_state
        interview_session.status = new_status
        return interview_session

    session_repository.update_if_unchanged.side_effect = persist_if_unchanged
    service = InterviewSessionService(session_repository, job_repository)
    return service, session_repository, job_repository


@pytest.mark.asyncio
async def test_create_session_initializes_state_from_job(
    service_dependencies: tuple[InterviewSessionService, AsyncMock, AsyncMock],
) -> None:
    """A new session stores the selected job context and starts in progress."""
    service, session_repository, job_repository = service_dependencies
    job_id = uuid4()
    job_repository.get_by_id.return_value = SimpleNamespace(
        id=job_id,
        description="Build Python backend services.",
    )

    session = await service.create_session(job_id)

    assert session.job_id == job_id
    assert session.status == InterviewSessionStatus.IN_PROGRESS
    assert session.state["job_id"] == str(job_id)
    assert session.state["job_description"] == "Build Python backend services."
    assert session.state["interview_history"] == []
    session_repository.create.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_create_session_rejects_unknown_job(
    service_dependencies: tuple[InterviewSessionService, AsyncMock, AsyncMock],
) -> None:
    """A session cannot be created without an existing job."""
    service, session_repository, job_repository = service_dependencies
    job_repository.get_by_id.return_value = None
    job_id = uuid4()

    with pytest.raises(JobNotFoundError):
        await service.create_session(job_id)

    session_repository.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_session_returns_validated_state(
    service_dependencies: tuple[InterviewSessionService, AsyncMock, AsyncMock],
) -> None:
    """An active session returns state ready for graph resumption."""
    service, session_repository, _ = service_dependencies
    job_id = uuid4()
    state = _initial_state(job_id)
    session = InterviewSession(
        job_id=job_id,
        status=InterviewSessionStatus.IN_PROGRESS,
        state={
            "job_id": str(job_id),
            "job_description": state["job_description"],
            "interview_history": [],
            "retrieved_context": [],
        },
    )
    session_repository.get_by_id.return_value = session

    loaded_session, restored_state = await service.resume_session(uuid4())

    assert loaded_session is session
    assert restored_state == state


@pytest.mark.asyncio
async def test_resume_session_rejects_completed_session(
    service_dependencies: tuple[InterviewSessionService, AsyncMock, AsyncMock],
) -> None:
    """A completed interview cannot be resumed as an active interview."""
    service, session_repository, _ = service_dependencies
    job_id = uuid4()
    state = _initial_state(job_id)
    state["interview_complete"] = True
    state["assessment_report"] = _assessment_report()
    session_repository.get_by_id.return_value = InterviewSession(
        job_id=job_id,
        status=InterviewSessionStatus.COMPLETED,
        state={
            **state,
            "job_id": str(job_id),
            "assessment_report": state["assessment_report"].model_dump(mode="json"),
        },
    )

    with pytest.raises(InterviewSessionAlreadyCompletedError):
        await service.resume_session(uuid4())


@pytest.mark.asyncio
async def test_save_state_marks_session_completed_with_report(
    service_dependencies: tuple[InterviewSessionService, AsyncMock, AsyncMock],
) -> None:
    """A final state is marked completed only when its report is present."""
    service, session_repository, _ = service_dependencies
    job_id = uuid4()
    session_id = uuid4()
    session = InterviewSession(
        id=session_id,
        job_id=job_id,
        status=InterviewSessionStatus.IN_PROGRESS,
        state={
            "job_id": str(job_id),
            "job_description": "Build Python backend services.",
            "interview_history": [],
            "retrieved_context": [],
        },
    )
    session_repository.get_by_id.return_value = session
    final_state = _initial_state(job_id)
    final_state["interview_complete"] = True
    final_state["assessment_report"] = _assessment_report()

    expected_state = dict(session.state)

    saved_session = await service.save_state(
        session_id,
        final_state,
        expected_state=expected_state,
    )

    assert saved_session.status == InterviewSessionStatus.COMPLETED
    assert saved_session.state["interview_complete"] is True
    assert "assessment_report" in saved_session.state
    session_repository.update_if_unchanged.assert_awaited_once_with(
        session,
        expected_state=expected_state,
        new_state=saved_session.state,
        new_status=InterviewSessionStatus.COMPLETED,
    )


@pytest.mark.asyncio
async def test_save_state_rejects_concurrent_modification(
    service_dependencies: tuple[InterviewSessionService, AsyncMock, AsyncMock],
) -> None:
    """A stale graph result must not overwrite a concurrently changed session."""
    service, session_repository, _ = service_dependencies
    job_id = uuid4()
    session_id = uuid4()

    session = InterviewSession(
        id=session_id,
        job_id=job_id,
        status=InterviewSessionStatus.IN_PROGRESS,
        state={
            "job_id": str(job_id),
            "job_description": "Build Python backend services.",
            "interview_history": [],
            "retrieved_context": [],
        },
    )
    session_repository.get_by_id.return_value = session

    # Simulate the atomic UPDATE finding no row matching the original snapshot.
    session_repository.update_if_unchanged.side_effect = None
    session_repository.update_if_unchanged.return_value = None

    expected_state = dict(session.state)

    with pytest.raises(InterviewSessionConflictError, match="changed"):
        await service.save_state(
            session_id,
            _initial_state(job_id),
            expected_state=expected_state,
        )

    session_repository.update_if_unchanged.assert_awaited_once()


@pytest.mark.asyncio
async def test_save_state_rejects_state_for_another_job(
    service_dependencies: tuple[InterviewSessionService, AsyncMock, AsyncMock],
) -> None:
    """State from another job cannot overwrite this session's state."""
    service, session_repository, _ = service_dependencies
    job_id = uuid4()
    session_id = uuid4()
    session_repository.get_by_id.return_value = InterviewSession(
        id=session_id,
        job_id=job_id,
        status=InterviewSessionStatus.IN_PROGRESS,
        state={
            "job_id": str(job_id),
            "job_description": "Build Python backend services.",
            "interview_history": [],
            "retrieved_context": [],
        },
    )

    expected_state = dict(session_repository.get_by_id.return_value.state)

    with pytest.raises(InvalidInterviewStateError, match="does not match"):
        await service.save_state(
            session_id,
            _initial_state(uuid4()),
            expected_state=expected_state,
        )

    session_repository.update_if_unchanged.assert_not_awaited()


@pytest.mark.asyncio
async def test_resume_session_rejects_missing_session(
    service_dependencies: tuple[InterviewSessionService, AsyncMock, AsyncMock],
) -> None:
    """A missing session produces a clear service-level error."""
    service, session_repository, _ = service_dependencies
    session_repository.get_by_id.return_value = None

    with pytest.raises(InterviewSessionNotFoundError):
        await service.resume_session(uuid4())
