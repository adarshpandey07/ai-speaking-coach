"""Baseline assessment — runs once, on a student's very first session.

It uses the same scorer as a regular session on purpose. If the baseline were
scored differently, "improvement since baseline" would be comparing two
different rulers.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from app.curriculum.weeks import BASELINE_PROMPTS, get_week
from app.mastery.tracker import ScoreBreakdown, Utterance, score_session
from app.models import BaselineAssessment

log = logging.getLogger(__name__)


def has_baseline(db: OrmSession, student_id: str) -> bool:
    return (
        db.execute(
            select(BaselineAssessment.id).where(
                BaselineAssessment.student_id == student_id
            )
        ).first()
        is not None
    )


def get_baseline(db: OrmSession, student_id: str) -> BaselineAssessment | None:
    return db.execute(
        select(BaselineAssessment).where(BaselineAssessment.student_id == student_id)
    ).scalar_one_or_none()


def score_baseline(utterances: list[Utterance]) -> ScoreBreakdown:
    # Week 1 targets: the baseline must not punish a beginner for brevity.
    return score_session(utterances, target_words_per_turn=get_week(1).target_words_per_turn)


def store_baseline(
    db: OrmSession, student_id: str, session_id: str, score: ScoreBreakdown
) -> BaselineAssessment:
    """Idempotent: a second call for the same student returns the existing row."""
    existing = get_baseline(db, student_id)
    if existing is not None:
        log.info("baseline already exists for student %s, not overwriting", student_id)
        return existing

    row = BaselineAssessment(
        student_id=student_id,
        session_id=session_id,
        overall=score.overall,
        fluency=score.fluency,
        vocabulary=score.vocabulary,
        elaboration=score.elaboration,
        clarity=score.clarity,
        notes=" ".join(score.notes),
    )
    db.add(row)
    db.flush()
    log.info("baseline stored for student %s: overall=%.1f", student_id, score.overall)
    return row


def prompt_for_turn(turn_index: int) -> str | None:
    """The fixed baseline prompt for a given turn, or None once they run out."""
    if 0 <= turn_index < len(BASELINE_PROMPTS):
        return BASELINE_PROMPTS[turn_index]
    return None
