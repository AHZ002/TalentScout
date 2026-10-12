"""SQLAlchemy model for a durable candidate interview session."""

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from talentscout.db.base import Base


class InterviewSessionStatus(StrEnum):
    """Lifecycle states for a candidate interview session."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class InterviewSession(Base):
    """Persist the complete LangGraph state for one candidate interview."""

    __tablename__ = "interview_sessions"

    id: Mapped[UUID] = mapped_column(
        primary_key=True,
        default=uuid4,
    )
    job_id: Mapped[UUID] = mapped_column(
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[InterviewSessionStatus] = mapped_column(
        String(20),
        default=InterviewSessionStatus.IN_PROGRESS,
        nullable=False,
    )
    state: Mapped[dict[str, object]] = mapped_column(
        JSONB,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
