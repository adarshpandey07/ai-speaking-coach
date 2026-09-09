"""Weeks 1-4 of the Phase 1 speaking-confidence curriculum.

Speaking and personality only — no academic-subject content. Each week is the
source of the coach's system prompt for that week, so the difficulty ramp lives
here rather than being buried in prompt strings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Week:
    week_number: int
    topic: str
    objective: str
    opener: str
    sample_prompts: list[str]
    difficulty_notes: str
    target_words_per_turn: int


WEEKS: list[Week] = [
    Week(
        week_number=1,
        topic="Talking about myself",
        objective=(
            "Say two or three connected sentences about yourself out loud "
            "without freezing. Getting words out matters more than getting "
            "them right."
        ),
        opener=(
            "Hi! I'm your speaking coach. There are no marks here and no wrong "
            "answers — we're just going to chat. So tell me, what's your name "
            "and what did you do today?"
        ),
        sample_prompts=[
            "What is your favourite food, and why do you like it?",
            "Tell me about someone in your family.",
            "What do you do after school every day?",
            "Which is your favourite subject? Say why in two sentences.",
        ],
        difficulty_notes=(
            "Easiest week. Short familiar topics. Accept one-word answers but "
            "always ask one warm follow-up to stretch the answer a little."
        ),
        target_words_per_turn=12,
    ),
    Week(
        week_number=2,
        topic="Describing things and places",
        objective=(
            "Describe a thing, a place or a picture using details and "
            "adjectives, in four or five sentences."
        ),
        opener=(
            "Good to see you back! Today we'll practise describing things. "
            "Close your eyes for a second and think of your favourite place — "
            "now tell me what it looks like."
        ),
        sample_prompts=[
            "Describe your classroom to someone who has never seen it.",
            "Describe the street outside your house in the evening.",
            "Tell me about a festival you enjoyed. What did you see and smell?",
            "Describe your favourite thing that you own.",
        ],
        difficulty_notes=(
            "Push for sensory detail and adjectives. If the student gives one "
            "sentence, ask 'what else did you see?' rather than moving on."
        ),
        target_words_per_turn=22,
    ),
    Week(
        week_number=3,
        topic="Telling a story",
        objective=(
            "Tell a short personal story with a beginning, a middle and an end, "
            "using past tense."
        ),
        opener=(
            "Today is story day. Everyone has stories — you just have to say "
            "them out loud. Tell me about something funny or surprising that "
            "happened to you recently."
        ),
        sample_prompts=[
            "Tell me about a day you will never forget.",
            "What happened the last time you were really excited?",
            "Tell me a story about you and your best friend.",
            "Describe a time something went wrong and what you did.",
        ],
        difficulty_notes=(
            "Watch past-tense verbs. Correct gently by repeating the sentence "
            "back correctly, never by saying 'that is wrong'."
        ),
        target_words_per_turn=32,
    ),
    Week(
        week_number=4,
        topic="Opinions and reasons",
        objective=(
            "State an opinion and back it with at least one reason, using "
            "phrases like 'I think', 'because' and 'in my opinion'."
        ),
        opener=(
            "You've been doing really well. Today we go one step harder — I "
            "want your opinion, and I want the reason behind it. Ready? "
            "Should students get homework every day?"
        ),
        sample_prompts=[
            "Is it better to play outside or play on a phone? Why?",
            "Should school start later in the morning? Give two reasons.",
            "What is the one thing you would change about your school?",
            "Do you think everyone should learn to cook? Why?",
        ],
        difficulty_notes=(
            "Hardest week of Phase 1. Always ask 'why do you think so?' at "
            "least once per turn. Disagree playfully to make them defend a view."
        ),
        target_words_per_turn=40,
    ),
]

WEEK_BY_NUMBER: dict[int, Week] = {w.week_number: w for w in WEEKS}
MAX_WEEK = max(WEEK_BY_NUMBER)


def get_week(week_number: int) -> Week:
    """Clamp out-of-range weeks rather than raising mid-session."""
    return WEEK_BY_NUMBER.get(min(max(week_number, 1), MAX_WEEK), WEEKS[0])


def as_rows() -> list[dict]:
    return [asdict(w) for w in WEEKS]


# --- Baseline assessment ------------------------------------------------------
# Fixed, deliberately easy prompts. They must stay constant across students and
# over time, otherwise "improvement vs baseline" is not comparable.
BASELINE_OPENER = (
    "Hi! Before we start our lessons, I want to hear you speak for a minute so "
    "I know how to help you. This is not a test and nothing is being marked. "
    "First question: tell me your name and one thing you like doing."
)

BASELINE_PROMPTS: list[str] = [
    "Tell me your name and one thing you like doing.",
    "Describe your best friend to me. What are they like?",
    "Tell me about something you did last weekend.",
]
