"""FastAPI app: HTTP for progress, WebSocket for the live spoken session.

One student turn over the socket:

    audio chunks -> ASR -> fallback check -> LangGraph agent -> safety filter
                 -> TTS -> audio back to the browser

The fallback check sits before the agent so a mis-heard utterance never reaches
the LLM, and the safety filter sits inside the graph so nothing reaches TTS
without passing it.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from app import store
from app.agent import fallback
from app.agent.graph import CoachAgent
from app.agent.llm import LLMConfigError, get_llm
from app.config import get_settings
from app.db import DatabaseUnavailable, check_connection, create_tables
from app.voice.asr import TranscriptResult, get_asr
from app.voice.tts import TTSError, get_tts

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
)
log = logging.getLogger("speaking-coach")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Postgres is not optional — fail now, with instructions, rather than at
    # the first student turn.
    check_connection()
    create_tables()
    await run_in_threadpool(store.seed_curriculum)

    # A missing API key should be visible before a child is sitting at the mic,
    # but it should not stop /health from explaining what is wrong.
    app.state.llm_ready = True
    app.state.llm_error = ""
    try:
        get_llm().preflight()
    except LLMConfigError as exc:
        app.state.llm_ready = False
        app.state.llm_error = str(exc)
        log.error("LLM is not configured:\n%s", exc)

    app.state.agent = await CoachAgent.create()

    # Load the Whisper model in the background so the first turn is not the one
    # that pays for it.
    async def _warm():
        try:
            await run_in_threadpool(get_asr().warmup)
            log.info("ASR model warm")
        except Exception as exc:  # model download can fail offline
            log.error("ASR warmup failed: %s", exc)

    app.state.warmup = asyncio.create_task(_warm())

    yield

    await app.state.agent.aclose()


app = FastAPI(title="AI Speaking Confidence Coach", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- HTTP --------------------------------------------------------------------


class StudentIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    grade: int = Field(default=6, ge=5, le=8)


@app.get("/api/health")
async def health() -> dict:
    db_ok, db_detail = True, "ok"
    try:
        await run_in_threadpool(check_connection)
    except DatabaseUnavailable as exc:
        db_ok, db_detail = False, str(exc)

    return {
        "status": "ok" if (db_ok and app.state.llm_ready) else "degraded",
        "database": {"ok": db_ok, "detail": db_detail},
        "llm": {
            "ok": app.state.llm_ready,
            "provider": settings.llm_provider,
            "model": settings.llm_model,
            "detail": app.state.llm_error or "ok",
        },
        "asr": {"provider": settings.asr_provider, "model": settings.asr_model_size},
        "tts": {"provider": settings.tts_provider, "voice": settings.tts_voice},
    }


@app.post("/api/students")
async def upsert_student(payload: StudentIn) -> dict:
    """No production auth for the MVP — a student is identified by name."""
    student = await run_in_threadpool(
        store.get_or_create_student, payload.name.strip(), payload.grade
    )
    await run_in_threadpool(store.seed_example_directive, student["id"])
    return student


@app.get("/api/students/{student_id}/progress")
async def student_progress(student_id: str) -> dict:
    try:
        return await run_in_threadpool(store.progress, student_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/students/{student_id}/sessions")
async def create_session(student_id: str) -> dict:
    try:
        return await run_in_threadpool(store.start_session, student_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/sessions/{session_id}/transcript")
async def session_transcript(session_id: str) -> list[dict]:
    return await run_in_threadpool(store.transcript, session_id)


@app.post("/api/sessions/{session_id}/complete")
async def finish_session(session_id: str, background: BackgroundTasks) -> dict:
    """Explicit end-of-session hook, for a client that closed without a socket."""
    background.add_task(_score_session, session_id)
    return {"accepted": True, "session_id": session_id}


# --- session plumbing --------------------------------------------------------


def _max_turns(mode: str) -> int:
    return settings.baseline_turns if mode == "baseline" else settings.session_max_turns


def _graph_context(ctx: dict) -> dict:
    return {
        "session_id": ctx["session_id"],
        "student_id": ctx["student_id"],
        "grade": ctx["grade"],
        "mode": ctx["mode"],
        "week_number": ctx["week_number"],
        "opener": ctx["opener"],
        "mastery": ctx["mastery"],
        "baseline": ctx["baseline"],
        "directive": ctx["directive"],
        "max_turns": _max_turns(ctx["mode"]),
    }


async def _score_session(session_id: str) -> None:
    """End-of-session scoring. Runs off the request/socket path."""
    try:
        result = await run_in_threadpool(store.complete_session, session_id)
        log.info("scored session %s: %s", session_id, result)
    except Exception:
        log.exception("failed to score session %s", session_id)


async def _speak(text: str) -> dict:
    """Render coach text to audio. Never raises — a silent turn still shows text."""
    try:
        speech = await get_tts().synthesize(text)
        return {
            "audio": base64.b64encode(speech.audio).decode("ascii"),
            "mime": speech.mime,
            "tts": speech.provider,
        }
    except (TTSError, Exception) as exc:
        log.error("TTS failed for %r: %s", text[:60], exc)
        return {"audio": None, "mime": None, "tts": None, "tts_error": str(exc)}


# --- WebSocket ---------------------------------------------------------------


@app.websocket("/ws/session/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    agent: CoachAgent = app.state.agent

    try:
        ctx = await run_in_threadpool(store.session_context, session_id)
    except LookupError:
        await websocket.send_json(
            {"type": "error", "code": "no_session", "message": "Unknown session id."}
        )
        await websocket.close(code=4404)
        return

    if not app.state.llm_ready:
        await websocket.send_json(
            {"type": "error", "code": "llm_unconfigured", "message": app.state.llm_error}
        )
        await websocket.close(code=4500)
        return

    context = _graph_context(ctx)
    max_turns = context["max_turns"]

    # A checkpoint with turns already on it means this socket is a reconnect,
    # not a fresh session: resume rather than replay the opener.
    snapshot = await agent.snapshot(session_id)
    resumed = bool(snapshot.get("turn_index"))
    turn_index = snapshot.get("turn_index", 0)

    history = await run_in_threadpool(store.transcript, session_id)
    ready = {
        "type": "ready",
        "session_id": session_id,
        "student_name": ctx["student_name"],
        "mode": ctx["mode"],
        "week_number": ctx["week_number"],
        "week_topic": ctx["week_topic"],
        "current_day": ctx["current_day"],
        "streak_days": ctx["streak_days"],
        "turn": turn_index,
        "max_turns": max_turns,
        "resumed": resumed,
        "transcript": history,
    }

    if resumed:
        log.info("session %s resumed at turn %s", session_id, turn_index)
        ready["coach"] = {"text": "Welcome back! We can carry on from where we stopped."}
        ready["coach"].update(await _speak(ready["coach"]["text"]))
    else:
        await agent.seed(session_id, context)
        ready["coach"] = {"text": ctx["opener"]}
        ready["coach"].update(await _speak(ctx["opener"]))

    await websocket.send_json(ready)

    audio_buffer = bytearray()
    consecutive_misses = 0
    scored = False

    try:
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                raise WebSocketDisconnect(message.get("code", 1000))

            if (chunk := message.get("bytes")) is not None:
                audio_buffer.extend(chunk)
                continue

            raw = message.get("text")
            if raw is None:
                continue
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                continue

            kind = event.get("type")

            if kind == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if kind == "end_session":
                await websocket.send_json({"type": "session_end", "reason": "student"})
                asyncio.create_task(_score_session(session_id))
                scored = True
                break

            if kind == "cancel_utterance":
                audio_buffer.clear()
                continue

            if kind != "end_utterance":
                continue

            # --- one student turn --------------------------------------------
            audio = bytes(audio_buffer)
            audio_buffer.clear()
            await websocket.send_json({"type": "thinking", "stage": "listening"})

            try:
                result: TranscriptResult = await run_in_threadpool(
                    get_asr().transcribe, audio, "audio/webm"
                )
            except Exception as exc:
                log.exception("ASR failed")
                await websocket.send_json(
                    {
                        "type": "error",
                        "code": "asr_failed",
                        "message": f"Could not process the audio: {exc}",
                    }
                )
                continue

            await websocket.send_json(
                {
                    "type": "transcript",
                    "text": result.text,
                    "confidence": result.confidence,
                    "seconds": round(result.duration_s, 2),
                }
            )

            decision = fallback.evaluate(
                result, settings.asr_min_confidence, consecutive_misses
            )
            if decision.triggered:
                consecutive_misses += 1
                log.info("fallback triggered (%s)", decision.reason)
                await run_in_threadpool(
                    store.add_turn,
                    session_id,
                    "coach",
                    decision.reprompt,
                    None,
                    None,
                    "fallback",
                )
                payload = {
                    "type": "coach",
                    "text": decision.reprompt,
                    "fallback": True,
                    "fallback_reason": decision.reason,
                    "turn": turn_index,
                    "max_turns": max_turns,
                    "finished": decision.give_up,
                }
                payload.update(await _speak(decision.reprompt))
                await websocket.send_json(payload)

                if decision.give_up:
                    await websocket.send_json(
                        {"type": "session_end", "reason": "audio_problem"}
                    )
                    asyncio.create_task(_score_session(session_id))
                    scored = True
                    break
                continue

            consecutive_misses = 0
            await run_in_threadpool(
                store.add_turn,
                session_id,
                "student",
                result.text,
                result.confidence,
                result.duration_s,
                None,
            )
            await websocket.send_json({"type": "thinking", "stage": "coach"})

            state = await agent.turn(session_id, result.text, context)
            reply = state.get("reply", "")
            turn_index = state.get("turn_index", turn_index + 1)
            finished = bool(state.get("finished"))

            await run_in_threadpool(
                store.add_turn,
                session_id,
                "coach",
                reply,
                None,
                None,
                state.get("safety_action"),
            )

            payload = {
                "type": "coach",
                "text": reply,
                "fallback": False,
                "turn": turn_index,
                "max_turns": max_turns,
                "finished": finished,
                "safety_action": state.get("safety_action"),
            }
            payload.update(await _speak(reply))
            await websocket.send_json(payload)

            if finished:
                await websocket.send_json(
                    {"type": "session_end", "reason": "turns_complete"}
                )
                # FastAPI-style background task: the socket can close now, the
                # mastery update and transcript write happen after.
                asyncio.create_task(_score_session(session_id))
                scored = True
                break

    except WebSocketDisconnect:
        # Not an error. The LangGraph checkpoint in Postgres holds the session,
        # so reopening this same session id resumes it.
        log.info("session %s disconnected at turn %s — checkpoint kept", session_id, turn_index)
        return
    except Exception:
        log.exception("session %s failed", session_id)
        try:
            await websocket.send_json(
                {
                    "type": "error",
                    "code": "internal",
                    "message": "Something went wrong. Reconnect to resume this session.",
                }
            )
        except Exception:
            pass
        return

    if scored:
        try:
            await websocket.close()
        except Exception:
            pass
