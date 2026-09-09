"""Age-appropriate output filter.

Every coach reply passes through here before it reaches TTS. Nothing that
fails reaches the child's speakers: the graph regenerates once, and if the
rewrite still fails it speaks a bland canned line instead.

The checks are deterministic rules rather than a second LLM call. For a
child-safety gate that runs on every single turn, a rule that always behaves
the same way is worth more than a cleverer one that sometimes does not — and it
adds no latency to a real-time conversation.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum


class SafetyAction(str, Enum):
    PASS = "pass"
    SANITIZED = "sanitized"  # cleaned up, safe to speak
    BLOCK = "block"  # must be regenerated or replaced


@dataclass
class SafetyVerdict:
    action: SafetyAction
    text: str
    reason: str = ""

    @property
    def ok(self) -> bool:
        return self.action is not SafetyAction.BLOCK


# Word-boundary matched, so "grass" never trips on "ass".
_BLOCKED_WORDS = [
    # profanity / slurs
    "fuck", "shit", "bitch", "bastard", "asshole", "slut", "whore", "dick",
    "penis", "vagina", "porn", "sex", "sexy", "naked", "nude",
    # violence / self-harm
    "kill", "suicide", "murder", "rape", "abuse", "blood", "weapon", "gun",
    "knife", "bomb", "die", "dead", "hurt yourself", "cut yourself",
    # substances
    "alcohol", "beer", "whisky", "drunk", "cigarette", "smoking", "drugs",
    "vape", "tobacco",
    # adult / relationship
    "girlfriend", "boyfriend", "dating", "kiss", "marriage proposal",
    # sensitive social topics for an unsupervised child session
    "caste", "religion", "hindu", "muslim", "christian", "politics",
    "election", "communal",
]

# Phrasings that would have the coach soliciting identifying information.
_PII_SOLICITATION = [
    r"\bwhat(?:'s| is) your (?:full |last |sur)?name\b",
    r"\bwhere do you live\b",
    r"\bwhat(?:'s| is) your (?:address|pin ?code|phone|mobile|number)\b",
    r"\bwhich (?:school|city|town|area|colony) (?:do you|are you)\b",
    r"\bname of your school\b",
    r"\bsend me (?:a )?(?:photo|picture|selfie)\b",
    r"\bmeet me\b",
    r"\btell me your parents?(?:'|s)? (?:name|number|phone)\b",
]

# Contact details must never be spoken to a child, whoever generated them.
_CONTACT = [
    r"https?://\S+",
    r"\bwww\.\S+\b",
    r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    r"(?<!\d)(?:\+91[\s-]?)?[6-9]\d{9}(?!\d)",
]

_BLOCKED_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in _BLOCKED_WORDS) + r")\b", re.IGNORECASE
)
_PII_RE = [re.compile(p, re.IGNORECASE) for p in _PII_SOLICITATION]
_CONTACT_RE = [re.compile(p, re.IGNORECASE) for p in _CONTACT]

MAX_SPOKEN_CHARS = 700


def sanitize_for_speech(text: str) -> str:
    """Strip anything the TTS voice would read out as noise.

    A stray '**' becomes "asterisk asterisk" in a child's ear, so formatting is
    removed rather than merely tolerated.
    """
    out = text.strip()
    out = re.sub(r"```.*?```", " ", out, flags=re.DOTALL)
    out = re.sub(r"[*_`#>]+", "", out)
    out = re.sub(r"^\s*[-•]\s*", "", out, flags=re.MULTILINE)
    out = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", out)  # markdown links
    out = re.sub(r"\((?:pauses?|laughs?|smiles?|warmly)[^)]*\)", "", out, flags=re.I)
    # Drop emoji and other symbol codepoints; keep normal punctuation.
    out = "".join(ch for ch in out if unicodedata.category(ch) not in {"So", "Cs"})
    out = re.sub(r"\s+", " ", out).strip()
    return out


def check(text: str) -> SafetyVerdict:
    """Screen one coach reply. Never raises — a blocked reply is a verdict."""
    if not text or not text.strip():
        return SafetyVerdict(SafetyAction.BLOCK, "", "empty reply")

    cleaned = sanitize_for_speech(text)
    if not cleaned:
        return SafetyVerdict(SafetyAction.BLOCK, "", "reply was only formatting")

    hit = _BLOCKED_RE.search(cleaned)
    if hit:
        return SafetyVerdict(
            SafetyAction.BLOCK, "", f"age-inappropriate word: {hit.group(1)!r}"
        )

    for pattern in _PII_RE:
        if pattern.search(cleaned):
            return SafetyVerdict(
                SafetyAction.BLOCK, "", "reply asks the student for personal details"
            )

    for pattern in _CONTACT_RE:
        if pattern.search(cleaned):
            return SafetyVerdict(
                SafetyAction.BLOCK, "", "reply contains a link, email or phone number"
            )

    if len(cleaned) > MAX_SPOKEN_CHARS:
        # Too long to speak, but not unsafe — trim at a sentence boundary.
        trimmed = cleaned[:MAX_SPOKEN_CHARS]
        cut = max(trimmed.rfind("."), trimmed.rfind("?"), trimmed.rfind("!"))
        cleaned = trimmed[: cut + 1] if cut > 200 else trimmed + "..."
        return SafetyVerdict(SafetyAction.SANITIZED, cleaned, "reply trimmed for length")

    if cleaned != text.strip():
        return SafetyVerdict(SafetyAction.SANITIZED, cleaned, "formatting stripped")

    return SafetyVerdict(SafetyAction.PASS, cleaned)
