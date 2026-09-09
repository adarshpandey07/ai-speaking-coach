"""What the coach says when it did not actually hear the student.

Silence, a noisy room and a low-confidence transcript are all normal in a real
classroom, and none of them should reach the LLM — a garbled transcript makes
the coach answer a question nobody asked, which is more confusing to a child
than simply being asked to repeat.

The re-prompts escalate: first a friendly "say that again", then concrete
advice, then a way out so the student is never stuck in a loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.voice.asr import TranscriptResult

# Below this many characters a "transcript" is almost always Whisper
# hallucinating on room noise ("you", "thanks", ".").
MIN_USEFUL_CHARS = 2

_NOISE_TRANSCRIPTS = {
    "you", "thank you", "thanks", "bye", "okay", ".", "...", "uh", "um",
    "thanks for watching", "thank you for watching",
}

_REPROMPTS = [
    "Sorry, I didn't quite catch that — can you say it again?",
    "I still couldn't hear you properly. Try speaking a bit louder, and a "
    "little closer to the microphone. What were you saying?",
    "No problem at all, let's make it easier. Just answer in one or two words "
    "— are you having a good day today?",
]

_GIVE_UP = (
    "I'm having trouble hearing you today. Check that your microphone is "
    "working, and come back and we'll pick up right here. You're doing fine!"
)

# After this many misses in a row we stop asking and suggest checking the mic.
MAX_CONSECUTIVE = 3


@dataclass
class FallbackDecision:
    triggered: bool
    reason: str = ""
    reprompt: str = ""
    give_up: bool = False


def evaluate(
    transcript: TranscriptResult,
    min_confidence: float,
    consecutive_failures: int = 0,
) -> FallbackDecision:
    """Decide whether this transcript is safe to send to the LLM."""
    text = transcript.text.strip()
    normalised = text.lower().strip(" .!?,")

    if transcript.is_silence or not text:
        reason = "silence"
    elif len(normalised) < MIN_USEFUL_CHARS or normalised in _NOISE_TRANSCRIPTS:
        reason = "noise-only transcript"
    elif transcript.confidence < min_confidence:
        reason = f"low ASR confidence ({transcript.confidence:.2f} < {min_confidence:.2f})"
    else:
        return FallbackDecision(triggered=False)

    if consecutive_failures >= MAX_CONSECUTIVE:
        return FallbackDecision(True, reason, _GIVE_UP, give_up=True)

    return FallbackDecision(True, reason, _REPROMPTS[min(consecutive_failures, len(_REPROMPTS) - 1)])
