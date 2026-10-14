"""FastAPI endpoints for starting and continuing candidate interviews."""

from collections.abc import AsyncIterator
from copy import deepcopy
from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from talentscout.agents.graph import build_interview_graph
from talentscout.agents.state import InterviewState
from talentscout.agents.state_serializer import deserialize_interview_state
from talentscout.db.models.interview_session import InterviewSession
from talentscout.db.session import get_session
from talentscout.documents.retriever import DocumentRetriever
from talentscout.embeddings.jina import JinaEmbeddingService
from talentscout.jobs.repositories.document_chunk import DocumentChunkRepository
from talentscout.jobs.repositories.interview_session import (
    InterviewSessionRepository,
)
from talentscout.jobs.repositories.job import JobRepository
from talentscout.jobs.schemas.interview_session import (
    InterviewAnswerSubmit,
    InterviewSessionResponse,
)
from talentscout.jobs.services.interview_session import (
    InterviewSessionAlreadyCompletedError,
    InterviewSessionConflictError,
    InterviewSessionNotFoundError,
    InterviewSessionService,
    JobNotFoundError,
)
from talentscout.llm.groq import GroqLLMService

router = APIRouter(tags=["interviews"])


def get_interview_session_service(
    session: AsyncSession = Depends(get_session),
) -> InterviewSessionService:
    """Build the interview-session service with request-scoped repositories."""
    return InterviewSessionService(
        session_repository=InterviewSessionRepository(session),
        job_repository=JobRepository(session),
    )


async def get_interview_graph(
    session: AsyncSession = Depends(get_session),
) -> AsyncIterator[Any]:
    """Build the real interview graph and close its LLM client after the request."""
    llm = GroqLLMService()

    try:
        embedding_service = JinaEmbeddingService()
        chunk_repository = DocumentChunkRepository(session)
        retriever = DocumentRetriever(
            repository=chunk_repository,
            embedding_service=embedding_service,
        )
        graph = build_interview_graph(retriever=retriever, llm=llm)

        yield graph
    finally:
        await llm.client.close()


def build_session_response(
    interview_session: InterviewSession,
    state: InterviewState,
) -> InterviewSessionResponse:
    """Convert validated persisted state into the public API response."""
    return InterviewSessionResponse(
        id=interview_session.id,
        job_id=interview_session.job_id,
        status=interview_session.status,
        current_question=state.get("current_question"),
        interview_complete=state.get("interview_complete", False),
        turn_count=len(state.get("interview_history", [])),
        assessment_report=state.get("assessment_report"),
    )


@router.post(
    "/jobs/{job_id}/interviews",
    response_model=InterviewSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_interview(
    job_id: UUID,
    service: InterviewSessionService = Depends(get_interview_session_service),
    graph: Any = Depends(get_interview_graph),
) -> InterviewSessionResponse:
    """Create a session and generate its first interview question."""
    try:
        interview_session = await service.create_session(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found",
        ) from exc

    interview_session, state = await service.resume_session(interview_session.id)

    # Preserve the exact database snapshot before the graph processes it.
    expected_state = deepcopy(interview_session.state)

    result = cast(InterviewState, await graph.ainvoke(state))

    try:
        saved_session = await service.save_state(
            interview_session.id,
            result,
            expected_state=expected_state,
        )
    except InterviewSessionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Interview session changed while processing. Please retry.",
        ) from exc

    return build_session_response(saved_session, result)


@router.get(
    "/interviews/{session_id}",
    response_model=InterviewSessionResponse,
)
async def get_interview(
    session_id: UUID,
    service: InterviewSessionService = Depends(get_interview_session_service),
) -> InterviewSessionResponse:
    """Return the current state of an interview session."""
    interview_session = await service.get_session(session_id)

    if interview_session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found",
        )

    state = deserialize_interview_state(interview_session.state)
    return build_session_response(interview_session, state)


@router.post(
    "/interviews/{session_id}/answers",
    response_model=InterviewSessionResponse,
)
async def submit_interview_answer(
    session_id: UUID,
    data: InterviewAnswerSubmit,
    service: InterviewSessionService = Depends(get_interview_session_service),
    graph: Any = Depends(get_interview_graph),
) -> InterviewSessionResponse:
    """Evaluate one candidate answer and return the updated interview state."""
    try:
        interview_session, state = await service.resume_session(session_id)
    except InterviewSessionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interview session not found",
        ) from exc
    except InterviewSessionAlreadyCompletedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Interview session is already completed",
        ) from exc

    # Capture the persisted state before modifying the graph input.
    expected_state = deepcopy(interview_session.state)

    state["candidate_answer"] = data.candidate_answer
    result = cast(InterviewState, await graph.ainvoke(state))

    try:
        saved_session = await service.save_state(
            interview_session.id,
            result,
            expected_state=expected_state,
        )
    except InterviewSessionConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Interview session was modified by another request. "
                "Retrieve the latest session state before submitting another answer."
            ),
        ) from exc

    return build_session_response(saved_session, result)
