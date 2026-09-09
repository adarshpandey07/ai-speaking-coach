"""SQLAlchemy models. Postgres is the only supported store."""

from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


class Student(Base):
    __tablename__ = "students"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    grade: Mapped[int] = mapped_column(Integer, default=6)
    # Curriculum position. Advances after a completed regular session.
    current_week: Mapped[int] = mapped_column(Integer, default=1)
    current_day: Mapped[int] = mapped_column(Integer, default=1)
    streak_days: Mapped[int] = mapped_column(Integer, default=0)
    last_session_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    sessions: Mapped[list[Session]] = relationship(back_populates="student")


class Session(Base):
    """One ~10 minute spoken conversation.

    `id` doubles as the LangGraph thread_id, which is what makes
    reconnect-and-resume work: the same id reopened over WebSocket finds its
    checkpoint in Postgres and continues instead of restarting.
    """

    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    mode: Mapped[str] = mapped_column(String(16), default="regular")  # baseline|regular
    week_number: Mapped[int] = mapped_column(Integer, default=1)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active|complete
    turn_count: Mapped[int] = mapped_column(Integer, default=0)
    # Denormalised snapshot written by the end-of-session background task.
    transcript: Mapped[list | None] = mapped_column(JSON, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    student: Mapped[Student] = relationship(back_populates="sessions")
    turns: Mapped[list[Turn]] = relationship(
        back_populates="session", order_by="Turn.index"
    )


class Turn(Base):
    """A single student utterance or coach reply."""

    __tablename__ = "turns"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), index=True)
    index: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))  # student|coach
    text: Mapped[str] = mapped_column(Text)
    asr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    audio_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Set when the safety filter rewrote or replaced a coach reply.
    safety_action: Mapped[str | None] = mapped_column(String(32), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    session: Mapped[Session] = relationship(back_populates="turns")


class BaselineAssessment(Base):
    """Scored once, on a student's very first session.

    Later "improvement" is measured against this row, so it is scored with the
    same rules a regular session is.
    """

    __tablename__ = "baseline_assessments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    student_id: Mapped[str] = mapped_column(
        ForeignKey("students.id"), unique=True, index=True
    )
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"))
    overall: Mapped[float] = mapped_column(Float)
    fluency: Mapped[float] = mapped_column(Float)
    vocabulary: Mapped[float] = mapped_column(Float)
    elaboration: Mapped[float] = mapped_column(Float)
    clarity: Mapped[float] = mapped_column(Float)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MasteryScore(Base):
    """One row per completed session, plus the rolling 5-session average."""

    __tablename__ = "mastery_scores"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), unique=True)
    overall: Mapped[float] = mapped_column(Float)
    fluency: Mapped[float] = mapped_column(Float)
    vocabulary: Mapped[float] = mapped_column(Float)
    elaboration: Mapped[float] = mapped_column(Float)
    clarity: Mapped[float] = mapped_column(Float)
    rolling_average: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class AdaptationDirective(Base):
    """A nudge from a parent or teacher, injected into the coach's system prompt.

    There is no UI to write these yet — the seeder inserts one inactive example
    row so the wiring is exercised end to end.
    """

    __tablename__ = "adaptation_directives"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    student_id: Mapped[str] = mapped_column(ForeignKey("students.id"), index=True)
    directive: Mapped[str] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), default="teacher")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class CurriculumWeek(Base):
    """Weeks 1-4 of the speaking-confidence curriculum, seeded on startup."""

    __tablename__ = "curriculum_weeks"

    week_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    topic: Mapped[str] = mapped_column(String(200))
    objective: Mapped[str] = mapped_column(Text)
    opener: Mapped[str] = mapped_column(Text)
    sample_prompts: Mapped[list] = mapped_column(JSON)
    difficulty_notes: Mapped[str] = mapped_column(Text)
    target_words_per_turn: Mapped[int] = mapped_column(Integer, default=20)
