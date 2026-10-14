"""Repository for durable candidate interview sessions."""

from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from talentscout.db.models.interview_session import (
    InterviewSession,
    InterviewSessionStatus,
)


class InterviewSessionRepository:
    """Handle database persistence for interview sessions."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the repository with a request-scoped database session."""
        self.session = session

    async def create(
        self,
        interview_session: InterviewSession,
    ) -> InterviewSession:
        """Persist a new interview session."""
        self.session.add(interview_session)
        await self.session.flush()
        await self.session.refresh(interview_session)
        return interview_session

    async def get_by_id(
        self,
        session_id: UUID,
    ) -> InterviewSession | None:
        """Find an interview session by its identifier."""
        result = await self.session.execute(
            select(InterviewSession).where(InterviewSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def update_if_unchanged(
        self,
        interview_session: InterviewSession,
        *,
        expected_state: dict[str, object],
        new_state: dict[str, object],
        new_status: InterviewSessionStatus,
    ) -> InterviewSession | None:
        """Update a session only if its persisted state has not changed.

        The conditional UPDATE is atomic. If another request has already
        changed the state or completed the interview, no row is updated.
        """
        result = await self.session.execute(
            update(InterviewSession)
            .where(
                InterviewSession.id == interview_session.id,
                InterviewSession.status == InterviewSessionStatus.IN_PROGRESS,
                InterviewSession.state == expected_state,
            )
            .values(
                state=new_state,
                status=new_status,
            )
            .returning(InterviewSession.id)
        )

        updated_id = result.scalar_one_or_none()

        if updated_id is None:
            return None

        # Refresh the existing ORM instance so callers receive the values
        # written by the database rather than its previous in-memory state.
        await self.session.refresh(interview_session)
        return interview_session
