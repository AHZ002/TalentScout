"""Integration tests for interview-session HTTP endpoints and persistence."""

from uuid import UUID, uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from talentscout.agents.assessment_report import AssessmentReport
from talentscout.agents.state import InterviewState
from talentscout.api.main import app
from talentscout.db.models.job import Job
from talentscout.db.session import SessionFactory
from talentscout.jobs.api.interviews import get_interview_graph


class FakeInterviewGraph:
    """Provide deterministic interview results without external providers."""

    def __init__(self, *, complete_on_answer: bool = False) -> None:
        self.calls: list[InterviewState] = []
        self.complete_on_answer = complete_on_answer

    async def ainvoke(self, state: InterviewState) -> InterviewState:
        """Record the input and return a predictable interview result."""
        self.calls.append({**state})
        result: InterviewState = {**state}

        if state.get("candidate_answer") and self.complete_on_answer:
            result["current_question"] = None
            result["candidate_answer"] = ""
            result["interview_complete"] = True
            result["assessment_report"] = AssessmentReport(
                overall_summary=("The candidate demonstrated strong Python fundamentals."),
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
        else:
            if state.get("candidate_answer"):
                result["current_question"] = "How would you test a FastAPI endpoint?"
                result["candidate_answer"] = ""
            else:
                result["current_question"] = "What is Python?"

            result["interview_complete"] = False

        result.setdefault("interview_history", [])
        result.setdefault("retrieved_context", [])
        return result


@pytest.mark.integration
@pytest.mark.asyncio
async def test_start_get_and_submit_interview() -> None:
    """Verify interview creation, retrieval, answer submission, and persistence."""
    fake_graph = FakeInterviewGraph()

    async def override_interview_graph() -> FakeInterviewGraph:
        """Return the deterministic graph used by this test."""
        return fake_graph

    previous_override = app.dependency_overrides.get(get_interview_graph)
    app.dependency_overrides[get_interview_graph] = override_interview_graph

    job_id: UUID | None = None

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            job_response = await client.post(
                "/jobs",
                json={
                    "title": "Python Backend Engineer",
                    "description": "Build backend services using Python and FastAPI.",
                },
            )
            assert job_response.status_code == 201
            job_id = UUID(job_response.json()["id"])

            start_response = await client.post(f"/jobs/{job_id}/interviews")
            assert start_response.status_code == 201

            started = start_response.json()
            session_id = UUID(started["id"])

            assert started["job_id"] == str(job_id)
            assert started["status"] == "in_progress"
            assert started["current_question"] == "What is Python?"
            assert started["interview_complete"] is False
            assert started["turn_count"] == 0

            get_response = await client.get(f"/interviews/{session_id}")
            assert get_response.status_code == 200
            assert get_response.json()["current_question"] == "What is Python?"

            answer = "Python is a high-level programming language."
            submit_response = await client.post(
                f"/interviews/{session_id}/answers",
                json={"candidate_answer": answer},
            )
            assert submit_response.status_code == 200

            submitted = submit_response.json()
            assert submitted["current_question"] == "How would you test a FastAPI endpoint?"
            assert submitted["status"] == "in_progress"
            assert submitted["interview_complete"] is False

            # Confirm the answer reached the graph and was consumed.
            assert fake_graph.calls[-1]["candidate_answer"] == answer

            # Confirm the updated question was persisted for later retrieval.
            persisted_response = await client.get(f"/interviews/{session_id}")
            assert persisted_response.status_code == 200
            assert (
                persisted_response.json()["current_question"]
                == "How would you test a FastAPI endpoint?"
            )
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_interview_graph, None)
        else:
            app.dependency_overrides[get_interview_graph] = previous_override

        if job_id is not None:
            async with SessionFactory() as session:
                await session.execute(delete(Job).where(Job.id == job_id))
                await session.commit()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interview_endpoints_handle_missing_resources() -> None:
    """Verify missing jobs and sessions return HTTP 404 responses."""
    fake_graph = FakeInterviewGraph()

    async def override_interview_graph() -> FakeInterviewGraph:
        """Return the deterministic graph used by this test."""
        return fake_graph

    previous_override = app.dependency_overrides.get(get_interview_graph)
    app.dependency_overrides[get_interview_graph] = override_interview_graph

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            missing_job_response = await client.post(f"/jobs/{uuid4()}/interviews")
            assert missing_job_response.status_code == 404

            missing_session_id = uuid4()

            get_response = await client.get(f"/interviews/{missing_session_id}")
            assert get_response.status_code == 404

            submit_response = await client.post(
                f"/interviews/{missing_session_id}/answers",
                json={"candidate_answer": "A valid answer."},
            )
            assert submit_response.status_code == 404

            invalid_answer_response = await client.post(
                f"/interviews/{missing_session_id}/answers",
                json={"candidate_answer": "   "},
            )
            assert invalid_answer_response.status_code == 422
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_interview_graph, None)
        else:
            app.dependency_overrides[get_interview_graph] = previous_override


@pytest.mark.integration
@pytest.mark.asyncio
async def test_completed_interview_persists_report_and_rejects_new_answers() -> None:
    """Verify completion, report persistence, and rejection of further answers."""
    fake_graph = FakeInterviewGraph(complete_on_answer=True)

    async def override_interview_graph() -> FakeInterviewGraph:
        """Return the deterministic graph used by this test."""
        return fake_graph

    previous_override = app.dependency_overrides.get(get_interview_graph)
    app.dependency_overrides[get_interview_graph] = override_interview_graph

    job_id: UUID | None = None

    try:
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            job_response = await client.post(
                "/jobs",
                json={
                    "title": "Python Backend Engineer",
                    "description": ("Build backend services using Python and FastAPI."),
                },
            )
            assert job_response.status_code == 201
            job_id = UUID(job_response.json()["id"])

            start_response = await client.post(f"/jobs/{job_id}/interviews")
            assert start_response.status_code == 201
            session_id = UUID(start_response.json()["id"])

            answer = "I would test endpoints using pytest and an HTTP client."
            completion_response = await client.post(
                f"/interviews/{session_id}/answers",
                json={"candidate_answer": answer},
            )
            assert completion_response.status_code == 200

            completed = completion_response.json()
            assert completed["status"] == "completed"
            assert completed["interview_complete"] is True
            assert completed["current_question"] is None
            assert completed["assessment_report"] is not None
            assert completed["assessment_report"]["overall_summary"] == (
                "The candidate demonstrated strong Python fundamentals."
            )

            # Verify the submitted answer reached the graph.
            assert fake_graph.calls[-1]["candidate_answer"] == answer

            # Verify the completed state and report survive a fresh GET.
            persisted_response = await client.get(f"/interviews/{session_id}")
            assert persisted_response.status_code == 200

            persisted = persisted_response.json()
            assert persisted["status"] == "completed"
            assert persisted["interview_complete"] is True
            assert persisted["assessment_report"] == completed["assessment_report"]

            # A completed session must not accept another answer or invoke
            # the graph again.
            graph_call_count = len(fake_graph.calls)
            rejected_response = await client.post(
                f"/interviews/{session_id}/answers",
                json={"candidate_answer": "Another answer."},
            )
            assert rejected_response.status_code == 409
            assert len(fake_graph.calls) == graph_call_count

            # Confirm the rejected request did not alter the saved report.
            final_response = await client.get(f"/interviews/{session_id}")
            assert final_response.status_code == 200
            assert final_response.json()["assessment_report"] == completed["assessment_report"]
            assert final_response.json()["status"] == "completed"
    finally:
        if previous_override is None:
            app.dependency_overrides.pop(get_interview_graph, None)
        else:
            app.dependency_overrides[get_interview_graph] = previous_override

        if job_id is not None:
            async with SessionFactory() as session:
                await session.execute(delete(Job).where(Job.id == job_id))
                await session.commit()
