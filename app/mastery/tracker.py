"""Rule-based speaking-mastery scoring.

This is a deliberate placeholder for a real pronunciation-assessment API. It
scores only what we can observe locally and cheaply: how much the student said,
how fast, how varied their words were, and how cleanly the recogniser heard
them. Every sub-score is 0-100 so the numbers stay comparable when a real
assessment API replaces the internals.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

# Spoken hesitation, as Whisper tends to render it.
HESITATION = {
    "um", "uh", "umm", "uhh", "hmm", "er", "err", "ah", "aa", "eh",
    "like", "actually", "basically", "means", "matlab", "haan", "achha",
}

# Roughly the 150 most common English words. Anything outside this is treated
# as "content vocabulary" — a crude but stable proxy for word variety.
COMMON = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it",
    "for", "not", "on", "with", "he", "as", "you", "do", "at", "this", "but",
    "his", "by", "from", "they", "we", "say", "her", "she", "or", "an",
    "will", "my", "one", "all", "would", "there", "their", "what", "so",
    "up", "out", "if", "about", "who", "get", "which", "go", "me", "when",
    "make", "can", "like", "time", "no", "just", "him", "know", "take",
    "people", "into", "year", "your", "good", "some", "could", "them", "see",
    "other", "than", "then", "now", "look", "only", "come", "its", "over",
    "think", "also", "back", "after", "use", "two", "how", "our", "work",
    "first", "well", "way", "even", "new", "want", "because", "any", "these",
    "give", "day", "most", "us", "is", "are", "was", "were", "am", "very",
    "much", "many", "here", "am", "did", "does", "had", "has", "been", "am",
    "yes", "yeah", "ok", "okay", "thing", "things", "really", "lot", "am",
}

_WORD = re.compile(r"[a-zA-Z']+")


@dataclass
class Utterance:
    """One student turn, as it was actually heard."""

    text: str
    asr_confidence: float = 1.0
    audio_seconds: float = 0.0

    @property
    def words(self) -> list[str]:
        return [w.lower() for w in _WORD.findall(self.text)]


@dataclass
class ScoreBreakdown:
    fluency: float = 0.0
    vocabulary: float = 0.0
    elaboration: float = 0.0
    clarity: float = 0.0
    overall: float = 0.0
    words_spoken: int = 0
    turns: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "fluency": self.fluency,
            "vocabulary": self.vocabulary,
            "elaboration": self.elaboration,
            "clarity": self.clarity,
            "overall": self.overall,
            "words_spoken": self.words_spoken,
            "turns": self.turns,
            "notes": self.notes,
        }


# Weights sum to 1.0. Elaboration and fluency dominate because Phase 1 is about
# getting a shy child to speak at length at all.
WEIGHTS = {"fluency": 0.30, "vocabulary": 0.20, "elaboration": 0.30, "clarity": 0.20}


def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return round(max(lo, min(hi, value)), 1)


def _band(value: float, low: float, high: float) -> float:
    """Map a value into 0-100, full marks anywhere inside [low, high]."""
    if value <= 0:
        return 0.0
    if value < low:
        return 100.0 * (value / low)
    if value <= high:
        return 100.0
    # Above the band, decay gently rather than cliff to zero.
    return 100.0 * math.exp(-(value - high) / high)


def score_session(
    utterances: list[Utterance], target_words_per_turn: int = 20
) -> ScoreBreakdown:
    real = [u for u in utterances if u.words]
    if not real:
        return ScoreBreakdown(notes=["No usable speech was captured."])

    all_words = [w for u in real for w in u.words]
    total_words = len(all_words)
    total_seconds = sum(u.audio_seconds for u in real)

    # --- Fluency: speaking rate, penalised by hesitation ---------------------
    if total_seconds > 0:
        wpm = total_words / (total_seconds / 60.0)
    else:
        wpm = float(target_words_per_turn) * 4  # neutral if no timing available
    hesitations = sum(1 for w in all_words if w in HESITATION)
    hesitation_ratio = hesitations / total_words
    fluency = _band(wpm, 70, 150) * (1.0 - min(hesitation_ratio * 2.0, 0.5))

    # --- Vocabulary: variety and content words -------------------------------
    unique_ratio = len(set(all_words)) / total_words
    content_ratio = sum(1 for w in set(all_words) if w not in COMMON) / len(
        set(all_words)
    )
    vocabulary = 100.0 * (0.55 * min(unique_ratio / 0.6, 1.0) + 0.45 * content_ratio)

    # --- Elaboration: is the student saying enough per turn? ------------------
    avg_per_turn = total_words / len(real)
    elaboration = _band(avg_per_turn, target_words_per_turn, target_words_per_turn * 3)

    # --- Clarity: how confidently the recogniser heard them -------------------
    mean_conf = sum(u.asr_confidence for u in real) / len(real)
    clarity = _clamp(mean_conf * 100.0)

    parts = {
        "fluency": _clamp(fluency),
        "vocabulary": _clamp(vocabulary),
        "elaboration": _clamp(elaboration),
        "clarity": clarity,
    }
    overall = _clamp(sum(parts[k] * WEIGHTS[k] for k in WEIGHTS))

    notes = []
    if hesitation_ratio > 0.12:
        notes.append("Frequent hesitation words — practise pausing silently instead.")
    if avg_per_turn < target_words_per_turn * 0.6:
        notes.append("Answers are shorter than this week's target.")
    if mean_conf < 0.7:
        notes.append("Speech was often hard to hear — encourage louder, slower speech.")
    if not notes:
        notes.append("Solid session — keep the same pace next time.")

    return ScoreBreakdown(
        **parts,
        overall=overall,
        words_spoken=total_words,
        turns=len(real),
        notes=notes,
    )


# Most recent session counts most. Five weights, newest first.
ROLLING_WEIGHTS = [5, 4, 3, 2, 1]


def rolling_average(recent_overalls: list[float]) -> float:
    """Weighted average of up to the last 5 session scores, newest first."""
    if not recent_overalls:
        return 0.0
    window = recent_overalls[:5]
    weights = ROLLING_WEIGHTS[: len(window)]
    return round(
        sum(s * w for s, w in zip(window, weights)) / sum(weights),
        1,
    )
