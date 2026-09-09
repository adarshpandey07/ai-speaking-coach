"""Speech recognition behind a swappable interface.

Every implementation must return a transcript *and* a confidence score. The
confidence is not decoration — app/agent/fallback.py uses it to decide whether
the transcript is trustworthy enough to send to the LLM at all.
"""

from __future__ import annotations

import io
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

from app.config import get_settings

log = logging.getLogger(__name__)


@dataclass
class TranscriptResult:
    text: str
    confidence: float  # 0.0 - 1.0
    duration_s: float = 0.0
    is_silence: bool = False


class ASRProvider(ABC):
    """Swap point for a hosted recogniser later."""

    @abstractmethod
    def transcribe(self, audio: bytes, mime: str = "audio/webm") -> TranscriptResult:
        """Decode `audio` and return what was said, with a confidence score."""


class FasterWhisperASR(ASRProvider):
    """Local Whisper via CTranslate2. No network, no API key.

    The model is downloaded once (~75 MB for base.en) into the HuggingFace
    cache and then loaded lazily, so app startup stays fast.
    """

    def __init__(self, model_size: str, device: str, compute_type: str) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            log.info("loading faster-whisper model %s (%s)", self._model_size, self._device)
            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def warmup(self) -> None:
        """Pay the model-load cost at startup instead of mid-conversation."""
        self._load()

    def transcribe(self, audio: bytes, mime: str = "audio/webm") -> TranscriptResult:
        if not audio:
            return TranscriptResult(text="", confidence=0.0, is_silence=True)

        model = self._load()
        # faster-whisper decodes container formats itself through PyAV, so the
        # browser's webm/opus blob can be handed over as-is.
        segments, info = model.transcribe(
            io.BytesIO(audio),
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        texts: list[str] = []
        logprobs: list[float] = []
        no_speech: list[float] = []
        for seg in segments:
            if seg.text.strip():
                texts.append(seg.text.strip())
                logprobs.append(seg.avg_logprob)
                no_speech.append(seg.no_speech_prob)

        duration = float(getattr(info, "duration", 0.0) or 0.0)
        if not texts:
            return TranscriptResult(
                text="", confidence=0.0, duration_s=duration, is_silence=True
            )

        mean_logprob = sum(logprobs) / len(logprobs)
        mean_no_speech = sum(no_speech) / len(no_speech)
        # exp(avg log-prob) is Whisper's own per-token likelihood; discount it by
        # how much of the clip the model thought was silence.
        confidence = math.exp(mean_logprob) * (1.0 - mean_no_speech)

        return TranscriptResult(
            text=" ".join(texts),
            confidence=round(max(0.0, min(1.0, confidence)), 3),
            duration_s=duration,
            is_silence=mean_no_speech > 0.7,
        )


@lru_cache
def get_asr() -> ASRProvider:
    s = get_settings()
    if s.asr_provider == "faster_whisper":
        return FasterWhisperASR(s.asr_model_size, s.asr_device, s.asr_compute_type)
    raise ValueError(f"Unknown ASR_PROVIDER: {s.asr_provider}")
