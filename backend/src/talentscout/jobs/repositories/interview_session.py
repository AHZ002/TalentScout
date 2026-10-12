"""Repository for durable candidate interview sessions."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from talentscout.db.models.interview_session import InterviewSession


class InterviewSessionRepository:
    """Handle database persistence for interview sessions."""

    def __init__(self, session: AsyncSession) -> None:
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

    async def update(
        self,
        interview_session: InterviewSession,
    ) -> InterviewSession:
        """Persist changes to an existing interview session."""
        await self.session.flush()
        await self.session.refresh(interview_session)
        return interview_session
