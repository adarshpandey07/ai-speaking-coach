"""Environment-driven configuration.

Everything the app talks to on the outside — the LLM, ASR, TTS, Postgres — is
named here and nowhere else, so a provider can be swapped by editing .env
rather than by editing code. There is deliberately no routing or fallback
between multiple LLMs: one provider, one model.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- LLM (single provider, single model) ---------------------------------
    # claude_cli  -> shells out to the local `claude` CLI (no API key needed)
    # anthropic   -> calls the Anthropic SDK directly with LLM_API_KEY
    llm_provider: Literal["claude_cli", "anthropic"] = "claude_cli"
    llm_model: str = "claude-sonnet-5"
    llm_api_key: str = ""
    llm_timeout_s: int = 90

    # --- ASR -----------------------------------------------------------------
    asr_provider: Literal["faster_whisper"] = "faster_whisper"
    asr_model_size: str = "base.en"
    asr_device: str = "cpu"
    asr_compute_type: str = "int8"
    # Below this, we do not trust the transcript and the fallback handler
    # generates a re-prompt instead of sending garbage to the LLM.
    asr_min_confidence: float = 0.55

    # --- TTS -----------------------------------------------------------------
    # edge_tts  -> free neural voices, needs network, no API key
    # macos_say -> fully offline, uses the built-in macOS `say` binary
    tts_provider: Literal["edge_tts", "macos_say"] = "edge_tts"
    tts_voice: str = "en-IN-NeerjaNeural"
    tts_macos_voice: str = "Samantha"
    # If the primary TTS raises, fall back to macos_say rather than killing the
    # session. This is a resilience path, not multi-model routing.
    tts_offline_fallback: bool = True

    # --- Storage -------------------------------------------------------------
    database_url: str = "postgresql://coach:coach@localhost:5432/speaking_coach"

    # --- Session shape -------------------------------------------------------
    session_max_turns: int = 10
    session_max_minutes: int = 10
    baseline_turns: int = 3

    @property
    def psycopg_dsn(self) -> str:
        """LangGraph's Postgres checkpointer wants a plain libpq DSN."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://")


@lru_cache
def get_settings() -> Settings:
    return Settings()
