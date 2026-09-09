"""The conversation agent: a LangGraph state graph over a Postgres checkpointer.

The checkpointer is the whole point of using a graph here. Session state lives
in Postgres keyed by `thread_id` (which is the session id), so when a
WebSocket drops mid-conversation — a phone leaving wifi, a laptop lid closing —
reopening the same session id restores the turn history and the conversation
continues instead of restarting from turn one.

Flow per student turn:  respond -> guard -> finalize
`guard` is not optional and cannot be skipped: nothing reaches TTS without
passing through the safety filter, and a blocked reply is regenerated once
before falling back to a canned safe line.
"""

from __future__ import annotations

import logging
import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from app.agent import prompts, safety
from app.agent.llm import LLMError, get_llm
from app.config import get_settings
from app.curriculum.weeks import get_week

log = logging.getLogger(__name__)


class CoachState(TypedDict, total=False):
    session_id: str
    student_id: str
    grade: int
    mode: str  # "baseline" | "regular"
    week_number: int
    opener: str
    mastery: dict | None
    baseline: dict | None
    directive: str | None
    max_turns: int

    messages: Annotated[list[dict], operator.add]
    turn_index: int

    # Per-turn scratch, overwritten each invocation.
    draft: str
    reply: str
    safety_action: str
    safety_reason: str
    llm_error: str
    finished: bool


def _system_prompt(state: CoachState) -> str:
    if state.get("mode") == "baseline":
        base = prompts.baseline_system_prompt(state.get("grade", 6))
    else:
        base = prompts.week_system_prompt(
            week=get_week(state.get("week_number", 1)),
            grade=state.get("grade", 6),
            mastery=state.get("mastery"),
            directive=state.get("directive"),
            baseline=state.get("baseline"),
        )

    opener = state.get("opener")
    if opener:
        # The opener is spoken without an LLM call, so the model is told about
        # it here rather than being handed an assistant-first message list.
        base += (
            f"\n\nYou have already greeted the student out loud with:\n"
            f'"{opener}"\nDo not greet them again.'
        )

    turn = state.get("turn_index", 0)
    max_turns = state.get("max_turns", 10)
    if turn >= max_turns - 1:
        base += (
            "\n\nThis is the last turn of the session. Praise one specific thing "
            "they did today, tell them you'll see them next time, and ask no "
            "further question."
        )
    return base


async def respond(state: CoachState) -> dict[str, Any]:
    """Call the LLM for this turn's coach reply."""
    system = _system_prompt(state)
    messages = [
        {"role": m["role"], "content": m["content"]} for m in state.get("messages", [])
    ]
    try:
        draft = await get_llm().complete(system, messages)
        return {"draft": draft, "llm_error": ""}
    except LLMError as exc:
        # A dead LLM must not kill the session; the guard node turns an empty
        # draft into the canned safe reply.
        log.error("LLM call failed: %s", exc)
        return {"draft": "", "llm_error": str(exc)}


async def guard(state: CoachState) -> dict[str, Any]:
    """Child-safety gate. Regenerate once, then fall back to a canned reply."""
    draft = state.get("draft", "")

    if not draft:
        return {
            "reply": prompts.SAFE_CANNED_REPLY,
            "safety_action": "canned",
            "safety_reason": state.get("llm_error") or "empty draft",
        }

    verdict = safety.check(draft)
    if verdict.ok:
        return {
            "reply": verdict.text,
            "safety_action": verdict.action.value,
            "safety_reason": verdict.reason,
        }

    log.warning("safety filter blocked a reply: %s", verdict.reason)
    system = _system_prompt(state) + prompts.REGENERATE_SUFFIX.format(
        reason=verdict.reason
    )
    messages = [
        {"role": m["role"], "content": m["content"]} for m in state.get("messages", [])
    ]
    try:
        retry = await get_llm().complete(system, messages)
    except LLMError as exc:
        log.error("regeneration failed: %s", exc)
        retry = ""

    second = safety.check(retry) if retry else None
    if second is not None and second.ok:
        return {
            "reply": second.text,
            "safety_action": "regenerated",
            "safety_reason": verdict.reason,
        }

    return {
        "reply": prompts.SAFE_CANNED_REPLY,
        "safety_action": "canned",
        "safety_reason": verdict.reason,
    }


async def finalize(state: CoachState) -> dict[str, Any]:
    """Commit the approved reply to history and decide whether we're done."""
    turn_index = state.get("turn_index", 0) + 1
    max_turns = state.get("max_turns", 10)
    return {
        "messages": [{"role": "assistant", "content": state["reply"]}],
        "turn_index": turn_index,
        "finished": turn_index >= max_turns,
        "draft": "",
    }


def build_graph():
    g = StateGraph(CoachState)
    g.add_node("respond", respond)
    g.add_node("guard", guard)
    g.add_node("finalize", finalize)
    g.set_entry_point("respond")
    g.add_edge("respond", "guard")
    g.add_edge("guard", "finalize")
    g.add_edge("finalize", END)
    return g


class CoachAgent:
    """Owns the compiled graph and the Postgres connection pool behind it."""

    def __init__(self, graph, pool) -> None:
        self._graph = graph
        self._pool = pool

    @classmethod
    async def create(cls) -> "CoachAgent":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool

        dsn = get_settings().psycopg_dsn
        pool = AsyncConnectionPool(
            conninfo=dsn,
            max_size=10,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await pool.open()
        saver = AsyncPostgresSaver(pool)
        await saver.setup()  # creates the checkpoint tables, idempotent
        graph = build_graph().compile(checkpointer=saver)
        log.info("LangGraph agent ready with Postgres checkpointer")
        return cls(graph, pool)

    async def aclose(self) -> None:
        await self._pool.close()

    @staticmethod
    def _config(session_id: str) -> dict:
        return {"configurable": {"thread_id": session_id}}

    async def turn(self, session_id: str, user_text: str, context: dict) -> CoachState:
        """Run one student turn through the graph and return the new state."""
        payload = {"messages": [{"role": "user", "content": user_text}], **context}
        return await self._graph.ainvoke(payload, config=self._config(session_id))

    async def snapshot(self, session_id: str) -> dict:
        """Restore a dropped session's state from the checkpointer."""
        state = await self._graph.aget_state(self._config(session_id))
        return dict(state.values) if state and state.values else {}

    async def seed(self, session_id: str, context: dict) -> None:
        """Record the spoken opener as turn zero without calling the LLM."""
        await self._graph.aupdate_state(self._config(session_id), context)
