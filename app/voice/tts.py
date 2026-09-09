"""Text to speech behind a swappable interface.

Two implementations ship: `edge_tts` (free neural voices, needs network, no API
key) and `macos_say` (fully offline). If the primary raises, we fall back to
the offline one rather than dropping the coach's turn on the floor — a silent
coach is indistinguishable from a crash to a 12-year-old.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from app.config import get_settings

log = logging.getLogger(__name__)


class TTSError(RuntimeError):
    pass


@dataclass
class SpeechResult:
    audio: bytes
    mime: str
    provider: str


class TTSProvider(ABC):
    @abstractmethod
    async def synthesize(self, text: str) -> SpeechResult:
        """Render `text` as speech audio the browser can play directly."""


class EdgeTTS(TTSProvider):
    """Microsoft Edge's free neural voices. No key, but needs connectivity."""

    def __init__(self, voice: str) -> None:
        self._voice = voice

    async def synthesize(self, text: str) -> SpeechResult:
        import edge_tts

        chunks = bytearray()
        try:
            communicate = edge_tts.Communicate(text, self._voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
        except Exception as exc:  # network, voice name, service hiccup
            raise TTSError(f"edge-tts failed: {exc}") from exc

        if not chunks:
            raise TTSError("edge-tts returned no audio")
        return SpeechResult(bytes(chunks), "audio/mpeg", "edge_tts")


class MacOSSayTTS(TTSProvider):
    """The `say` binary shipped with macOS. Works with the network unplugged."""

    def __init__(self, voice: str) -> None:
        self._voice = voice

    async def synthesize(self, text: str) -> SpeechResult:
        def _run() -> bytes:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "speech.wav"
                proc = subprocess.run(
                    [
                        "say",
                        "-v", self._voice,
                        "--data-format=LEI16@22050",
                        "--file-format=WAVE",
                        "-o", str(out),
                        text,
                    ],
                    capture_output=True,
                    timeout=60,
                )
                if proc.returncode != 0 or not out.exists():
                    raise TTSError(
                        f"macOS say failed: {proc.stderr.decode(errors='replace')}"
                    )
                return out.read_bytes()

        audio = await asyncio.to_thread(_run)
        return SpeechResult(audio, "audio/wav", "macos_say")


class FallbackTTS(TTSProvider):
    """Primary provider, with the offline one as a safety net."""

    def __init__(self, primary: TTSProvider, backup: TTSProvider) -> None:
        self._primary = primary
        self._backup = backup

    async def synthesize(self, text: str) -> SpeechResult:
        try:
            return await self._primary.synthesize(text)
        except Exception as exc:
            log.warning("primary TTS failed (%s) — using offline fallback", exc)
            return await self._backup.synthesize(text)


def _build(name: str, s) -> TTSProvider:
    if name == "edge_tts":
        return EdgeTTS(s.tts_voice)
    if name == "macos_say":
        return MacOSSayTTS(s.tts_macos_voice)
    raise ValueError(f"Unknown TTS_PROVIDER: {name}")


@lru_cache
def get_tts() -> TTSProvider:
    s = get_settings()
    primary = _build(s.tts_provider, s)
    if s.tts_offline_fallback and s.tts_provider != "macos_say":
        return FallbackTTS(primary, MacOSSayTTS(s.tts_macos_voice))
    return primary
