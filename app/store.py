"""Synchronous data access, kept out of main.py so the WebSocket handler stays
readable.

Every function here is blocking. Async callers wrap them in
`starlette.concurrency.run_in_threadpool`.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from app.curriculum.weeks import BASELINE_OPENER, MAX_WEEK, as_rows, get_week
from app.db import db_session
from app.mastery import baseline as baseline_mod
from app.mastery.tracker import Utterance, rolling_average, score_session
from app.models import (
    AdaptationDirective,
    CurriculumWeek,
    MasteryScore,
    Session,
    Student,
    Turn,
)

log = logging.getLogger(__name__)

DAYS_PER_WEEK = 5


# --- seeding -----------------------------------------------------------------

def seed_curriculum() -> None:
    with db_session() as db:
        for row in as_rows():
            existing = db.get(CurriculumWeek, row["week_number"])
            if existing is None:
                db.add(CurriculumWeek(**row))
            else:
                for key, value in row.items():
                    setattr(existing, key, value)
    log.info("curriculum weeks 1-%s seeded", MAX_WEEK)


def seed_example_directive(student_id: str) -> None:
    """There is no parent/teacher UI yet, so one inactive example row is
    inserted to prove the column is wired into the prompt. Flip `active` to
    true in psql to see it take effect."""
    with db_session() as db:
        exists = db.execute(
            select(AdaptationDirective.id).where(
                AdaptationDirective.student_id == student_id
            )
        ).first()
        if exists:
            return
        db.add(
            AdaptationDirective(
                student_id=student_id,
                directive=(
                    "She is very shy about speaking in front of others. Please "
                    "give extra encouragement and keep questions simple."
                ),
                source="teacher",
                active=False,
            )
        )


# --- students ----------------------------------------------------------------

def get_or_create_student(name: str, grade: int = 6) -> dict:
    with db_session() as db:
        student = db.execute(
            select(Student).where(Student.name == name)
        ).scalar_one_or_none()
        if student is None:
            student = Student(name=name, grade=grade)
            db.add(student)
            db.flush()
            log.info("created student %s (%s)", name, student.id)
        return _student_dict(student)


def _student_dict(s: Student) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "grade": s.grade,
        "current_week": s.current_week,
        "current_day": s.current_day,
        "streak_days": s.streak_days,
    }


def progress(student_id: str) -> dict:
    with db_session() as db:
        student = db.get(Student, student_id)
        if student is None:
            raise LookupError(f"no student {student_id}")
        out = _student_dict(student)
        out["baseline"] = _baseline_dict(db, student_id)
        out["mastery"] = _latest_mastery(db, student_id)
        out["sessions_completed"] = (
            db.query(MasteryScore).filter_by(student_id=student_id).count()
        )
        week = get_week(student.current_week)
        out["week_topic"] = week.topic
        out["week_objective"] = week.objective
        return out


def _baseline_dict(db: OrmSession, student_id: str) -> dict | None:
    row = baseline_mod.get_baseline(db, student_id)
    if row is None:
        return None
    return {
        "overall": row.overall,
        "fluency": row.fluency,
        "vocabulary": row.vocabulary,
        "elaboration": row.elaboration,
        "clarity": row.clarity,
    }


def _latest_mastery(db: OrmSession, student_id: str) -> dict | None:
    row = db.execute(
        select(MasteryScore)
        .where(MasteryScore.student_id == student_id)
        .order_by(MasteryScore.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    return {
        "overall": row.overall,
        "fluency": row.fluency,
        "vocabulary": row.vocabulary,
        "elaboration": row.elaboration,
        "clarity": row.clarity,
        "rolling_average": row.rolling_average,
    }


def active_directive(db: OrmSession, student_id: str) -> str | None:
    row = db.execute(
        select(AdaptationDirective)
        .where(
            AdaptationDirective.student_id == student_id,
            AdaptationDirective.active.is_(True),
        )
        .order_by(AdaptationDirective.created_at.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row.directive if row else None


# --- sessions ----------------------------------------------------------------

def start_session(student_id: str) -> dict:
    """Create a session and return everything the WebSocket needs to open it.

    A student with no BaselineAssessment row gets a baseline session; everyone
    else gets the regular weekly curriculum.
    """
    with db_session() as db:
        student = db.get(Student, student_id)
        if student is None:
            raise LookupError(f"no student {student_id}")

        is_baseline = not baseline_mod.has_baseline(db, student_id)
        mode = "baseline" if is_baseline else "regular"
        week = get_week(student.current_week)
        opener = BASELINE_OPENER if is_baseline else week.opener

        session = Session(
            student_id=student_id,
            mode=mode,
            week_number=student.current_week,
            status="active",
        )
        db.add(session)
        db.flush()

        # The opener is spoken, so it is a real turn in the transcript even
        # though no LLM call produced it.
        db.add(Turn(session_id=session.id, index=0, role="coach", text=opener))
        session.turn_count = 0

        return {
            "session_id": session.id,
            "student": _student_dict(student),
            "mode": mode,
            "week_number": session.week_number,
            "week_topic": week.topic,
            "opener": opener,
            "baseline": _baseline_dict(db, student_id),
            "mastery": _latest_mastery(db, student_id),
            "directive": active_directive(db, student_id),
        }


def session_context(session_id: str) -> dict:
    """Everything the graph needs, refreshed on every (re)connect."""
    with db_session() as db:
        session = db.get(Session, session_id)
        if session is None:
            raise LookupError(f"no session {session_id}")
        student = db.get(Student, session.student_id)
        week = get_week(session.week_number)
        return {
            "session_id": session.id,
            "student_id": student.id,
            "student_name": student.name,
            "grade": student.grade,
            "mode": session.mode,
            "week_number": session.week_number,
            "week_topic": week.topic,
            "status": session.status,
            "turn_count": session.turn_count,
            "opener": next(
                (t.text for t in session.turns if t.index == 0 and t.role == "coach"),
                week.opener,
            ),
            "mastery": _latest_mastery(db, student.id),
            "baseline": _baseline_dict(db, student.id),
            "directive": active_directive(db, student.id),
            "streak_days": student.streak_days,
            "current_day": student.current_day,
        }


def transcript(session_id: str) -> list[dict]:
    with db_session() as db:
        session = db.get(Session, session_id)
        if session is None:
            return []
        return [
            {
                "index": t.index,
                "role": t.role,
                "text": t.text,
                "confidence": t.asr_confidence,
                "safety_action": t.safety_action,
            }
            for t in session.turns
        ]


def add_turn(
    session_id: str,
    role: str,
    text: str,
    asr_confidence: float | None = None,
    audio_seconds: float | None = None,
    safety_action: str | None = None,
) -> int:
    with db_session() as db:
        session = db.get(Session, session_id)
        next_index = (
            db.query(Turn).filter_by(session_id=session_id).count()
        )
        db.add(
            Turn(
                session_id=session_id,
                index=next_index,
                role=role,
                text=text,
                asr_confidence=asr_confidence,
                audio_seconds=audio_seconds,
                safety_action=safety_action,
            )
        )
        if session is not None and role == "student":
            session.turn_count = session.turn_count + 1
        return next_index


# --- end of session ----------------------------------------------------------

def complete_session(session_id: str) -> dict:
    """Score the session, update mastery/streak/curriculum position.

    Runs as a FastAPI background task after the socket closes, so a slow write
    never delays the student's last spoken reply.
    """
    with db_session() as db:
        session = db.get(Session, session_id)
        if session is None:
            raise LookupError(f"no session {session_id}")
        if session.status == "complete":
            log.info("session %s already scored", session_id)
            return {"already_complete": True}

        student = db.get(Student, session.student_id)
        week = get_week(session.week_number)

        student_turns = [t for t in session.turns if t.role == "student"]
        utterances = [
            Utterance(
                text=t.text,
                asr_confidence=t.asr_confidence if t.asr_confidence is not None else 1.0,
                audio_seconds=t.audio_seconds or 0.0,
            )
            for t in student_turns
        ]

        session.transcript = [
            {"index": t.index, "role": t.role, "text": t.text} for t in session.turns
        ]
        session.status = "complete"
        session.ended_at = datetime.utcnow()

        if not utterances:
            log.info("session %s had no student speech — not scored", session_id)
            return {"scored": False, "reason": "no student speech captured"}

        if session.mode == "baseline":
            score = baseline_mod.score_baseline(utterances)
            baseline_mod.store_baseline(db, student.id, session.id, score)
            result = {"scored": True, "mode": "baseline", **score.as_dict()}
        else:
            score = score_session(utterances, week.target_words_per_turn)
            history = [
                row.overall
                for row in db.execute(
                    select(MasteryScore)
                    .where(MasteryScore.student_id == student.id)
                    .order_by(MasteryScore.created_at.desc())
                    .limit(4)
                ).scalars()
            ]
            rolling = rolling_average([score.overall, *history])
            db.add(
                MasteryScore(
                    student_id=student.id,
                    session_id=session.id,
                    overall=score.overall,
                    fluency=score.fluency,
                    vocabulary=score.vocabulary,
                    elaboration=score.elaboration,
                    clarity=score.clarity,
                    rolling_average=rolling,
                )
            )
            result = {
                "scored": True,
                "mode": "regular",
                "rolling_average": rolling,
                **score.as_dict(),
            }
            _advance_curriculum(student)

        _update_streak(student)
        log.info("session %s complete: %s", session_id, result)
        return result


def _update_streak(student: Student) -> None:
    today = date.today()
    last = student.last_session_on
    if last == today:
        pass  # already counted today
    elif last == today - timedelta(days=1):
        student.streak_days += 1
    else:
        student.streak_days = 1
    student.last_session_on = today


def _advance_curriculum(student: Student) -> None:
    if student.current_day >= DAYS_PER_WEEK:
        student.current_day = 1
        student.current_week = min(student.current_week + 1, MAX_WEEK)
    else:
        student.current_day += 1
