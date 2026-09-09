"""The single LLM provider, called directly through its own SDK.

There is no LiteLLM and no routing layer: `LLM_PROVIDER` picks exactly one
implementation at startup and that is the one used for the whole process. The
interface is two lines wide so a third provider can be dropped in later without
touching the graph.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from functools import lru_cache

from app.config import get_settings

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMConfigError(LLMError):
    """Raised at startup for a missing key, so it never surfaces mid-session."""


class LLMClient(ABC):
    name: str

    @abstractmethod
    async def complete(self, system: str, messages: list[dict], max_tokens: int = 400) -> str:
        """Return the assistant's reply text for `messages` under `system`."""

    def preflight(self) -> None:
        """Fail loudly at startup if this provider can never work."""


class AnthropicLLM(LLMClient):
    """Direct Anthropic SDK call. Requires LLM_API_KEY."""

    name = "anthropic"

    def __init__(self, model: str, api_key: str, timeout_s: int) -> None:
        self._model = model
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._client = None

    def preflight(self) -> None:
        if not self._api_key:
            raise LLMConfigError(
                "LLM_PROVIDER=anthropic needs LLM_API_KEY in your .env.\n"
                "Either add the key, or switch to the keyless local path:\n"
                "  LLM_PROVIDER=claude_cli"
            )

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic

            self._client = AsyncAnthropic(api_key=self._api_key, timeout=self._timeout_s)
        return self._client

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 400) -> str:
        client = self._get_client()
        try:
            resp = await client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
            )
        except Exception as exc:
            raise LLMError(f"Anthropic call failed: {exc}") from exc

        parts = [block.text for block in resp.content if getattr(block, "type", "") == "text"]
        text = "".join(parts).strip()
        if not text:
            raise LLMError("Anthropic returned an empty response")
        return text


class ClaudeCLILLM(LLMClient):
    """Shells out to the locally installed `claude` CLI — no API key needed.

    The CLI is stateless in `-p` mode, so the conversation LangGraph is holding
    gets rendered into the prompt. Tools and MCP are switched off: this process
    only ever wants text back, and every tool left enabled is latency and
    prompt budget spent for nothing.
    """

    name = "claude_cli"

    DISABLED_TOOLS = (
        "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite,NotebookEdit"
    )

    def __init__(self, model: str, timeout_s: int) -> None:
        self._model = model
        self._timeout_s = timeout_s

    def preflight(self) -> None:
        from shutil import which

        if which("claude") is None:
            raise LLMConfigError(
                "LLM_PROVIDER=claude_cli but the `claude` binary is not on PATH.\n"
                "Install Claude Code, or switch to LLM_PROVIDER=anthropic with an "
                "LLM_API_KEY."
            )

    @staticmethod
    def _render(messages: list[dict]) -> str:
        """Flatten the turn history into one prompt the stateless CLI can read."""
        lines = []
        for m in messages:
            speaker = "Student" if m["role"] == "user" else "Coach"
            lines.append(f"{speaker}: {m['content']}")
        lines.append("Coach:")
        return "\n".join(lines)

    async def complete(self, system: str, messages: list[dict], max_tokens: int = 400) -> str:
        prompt = self._render(messages)
        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", "json",
            "--model", self._model,
            "--append-system-prompt", system,
            "--strict-mcp-config",
            "--disallowed-tools", self.DISABLED_TOOLS,
        ]
        # Run from an empty directory so no project's CLAUDE.md or settings leak
        # into a child's tutoring session.
        with tempfile.TemporaryDirectory() as cwd:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
                )
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout_s
                )
            except asyncio.TimeoutError as exc:
                raise LLMError(f"claude CLI timed out after {self._timeout_s}s") from exc
            except FileNotFoundError as exc:
                raise LLMConfigError("`claude` binary not found on PATH") from exc

        if proc.returncode != 0:
            raise LLMError(
                f"claude CLI exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}"
            )
        try:
            payload = json.loads(stdout.decode())
        except json.JSONDecodeError as exc:
            raise LLMError(f"claude CLI returned non-JSON: {stdout[:300]!r}") from exc

        if payload.get("is_error"):
            raise LLMError(f"claude CLI reported an error: {payload.get('result')}")
        text = (payload.get("result") or "").strip()
        if not text:
            raise LLMError("claude CLI returned an empty result")
        return text


@lru_cache
def get_llm() -> LLMClient:
    s = get_settings()
    if s.llm_provider == "anthropic":
        client: LLMClient = AnthropicLLM(s.llm_model, s.llm_api_key, s.llm_timeout_s)
    elif s.llm_provider == "claude_cli":
        client = ClaudeCLILLM(s.llm_model, s.llm_timeout_s)
    else:  # pragma: no cover - pydantic already constrains this
        raise LLMConfigError(f"Unknown LLM_PROVIDER: {s.llm_provider}")
    return client
