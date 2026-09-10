#!/usr/bin/env python
"""End-to-end check for a running local stack.

Drives the app the way the browser does — synthesising a child's voice with
edge-tts, streaming it over the WebSocket in chunks, and asserting on what
comes back. It exercises the three behaviours that are easy to break and
impossible to eyeball: the day-one baseline, reconnect-and-resume, and the
low-confidence fallback.

    # backend must already be running
    uvicorn app.main:app --port 8010
    python scripts/e2e_check.py

Exits non-zero on the first failed assertion.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import urllib.request
import uuid

import edge_tts
import websockets

BASE = os.environ.get("COACH_BASE", "http://127.0.0.1:8010")
WS = BASE.replace("http://", "ws://").replace("https://", "wss://")
# A child voice, so the recogniser sees input shaped like the real thing.
STUDENT_VOICE = os.environ.get("COACH_TEST_VOICE", "en-US-AnaNeural")
RECV_TIMEOUT = 120


def post(path: str, body: dict | None = None) -> dict:
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(body or {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return json.load(urllib.request.urlopen(req))


def get(path: str) -> dict:
    return json.load(urllib.request.urlopen(BASE + path))


async def say(text: str) -> bytes:
    out = bytearray()
    async for chunk in edge_tts.Communicate(text, STUDENT_VOICE).stream():
        if chunk["type"] == "audio":
            out.extend(chunk["data"])
    return bytes(out)


async def recv(ws) -> dict:
    return json.loads(await asyncio.wait_for(ws.recv(), timeout=RECV_TIMEOUT))


async def wait_ready(ws) -> dict:
    while True:
        msg = await recv(ws)
        if msg["type"] == "ready":
            return msg
        if msg["type"] == "error":
            raise AssertionError(f"server error before ready: {msg}")


async def speak_turn(ws, text: str) -> tuple[dict | None, dict]:
    """Stream one utterance the way MediaRecorder does, then read the reply."""
    audio = await say(text)
    for i in range(0, len(audio), 4096):
        await ws.send(audio[i : i + 4096])
    await ws.send(json.dumps({"type": "end_utterance"}))

    heard = None
    while True:
        msg = await recv(ws)
        if msg["type"] == "transcript":
            heard = msg
        elif msg["type"] == "coach":
            return heard, msg
        elif msg["type"] == "session_end":
            return heard, {"text": "(session ended)", "finished": True, "turn": -1}
        elif msg["type"] == "error":
            raise AssertionError(f"server error mid-turn: {msg}")


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


async def test_baseline(student: dict) -> None:
    banner("TEST 1 — day-one baseline session")
    sess = post(f"/api/students/{student['id']}/sessions")
    assert sess["mode"] == "baseline", "a student with no baseline must get one"
    print(f"session={sess['session_id'][:8]} mode={sess['mode']}")

    answers = [
        "My name is Anya and I like playing badminton with my brother in the evening.",
        "My best friend is Meera. She is very funny and she always shares her lunch with me.",
        "Last weekend I went to my grandmother's house and we made mango pickle together.",
    ]
    async with websockets.connect(
        f"{WS}/ws/session/{sess['session_id']}", max_size=None
    ) as ws:
        ready = await wait_ready(ws)
        assert ready["coach"].get("audio"), "the opener was not spoken"
        print(f"  opener via {ready['coach'].get('tts')}: {ready['coach']['text'][:80]}")

        for answer in answers:
            started = time.time()
            heard, coach = await speak_turn(ws, answer)
            assert heard is not None, "no transcript came back"
            print(f"\n  said  : {answer[:66]}")
            print(f"  heard : {heard['text'][:66]!r} conf={heard['confidence']}")
            print(
                f"  coach : ({time.time() - started:.1f}s "
                f"safety={coach.get('safety_action')} "
                f"audio={'yes' if coach.get('audio') else 'no'}) {coach['text'][:88]}"
            )
            if coach.get("finished"):
                break

    await asyncio.sleep(4)  # scoring is a background task
    baseline = get(f"/api/students/{student['id']}/progress")["baseline"]
    assert baseline, "BaselineAssessment was not persisted"
    print(f"\n  baseline stored: {baseline}")


async def test_resume(student: dict) -> None:
    banner("TEST 2 — reconnect-and-resume across a dropped socket")
    sess = post(f"/api/students/{student['id']}/sessions")
    assert sess["mode"] == "regular", "the second session should follow the curriculum"
    print(f"session={sess['session_id'][:8]} week={sess['week_number']}")

    ws = await websockets.connect(f"{WS}/ws/session/{sess['session_id']}", max_size=None)
    first = await wait_ready(ws)
    assert first["resumed"] is False
    _, coach = await speak_turn(
        ws, "I like reading story books because they take me to new places."
    )
    before = coach["turn"]
    await ws.close()  # simulate wifi dropping out
    print(f"  turn {before} done, socket dropped abruptly")
    await asyncio.sleep(1)

    async with websockets.connect(
        f"{WS}/ws/session/{sess['session_id']}", max_size=None
    ) as ws2:
        again = await wait_ready(ws2)
        assert again["resumed"] is True, "session did not resume from the checkpointer"
        assert again["turn"] == before, "turn index was lost across the reconnect"
        print(f"  reconnected: turn={again['turn']} history={len(again['transcript'])} turns")

        _, coach2 = await speak_turn(
            ws2, "My favourite place is the terrace because I can see the whole street."
        )
        assert coach2["turn"] == before + 1, "conversation restarted instead of resuming"
        print(f"  continued at turn {coach2['turn']}: {coach2['text'][:80]}")
        await ws2.send(json.dumps({"type": "end_session"}))
        await asyncio.sleep(1)


async def test_fallback(student: dict) -> None:
    banner("TEST 3 — fallback handler on unusable audio")
    sess = post(f"/api/students/{student['id']}/sessions")
    async with websockets.connect(
        f"{WS}/ws/session/{sess['session_id']}", max_size=None
    ) as ws:
        await wait_ready(ws)
        silence = await say(" ")
        await ws.send(silence[:200])  # truncated: nothing decodable in it
        await ws.send(json.dumps({"type": "end_utterance"}))

        while True:
            msg = await recv(ws)
            if msg["type"] == "coach":
                assert msg["fallback"] is True, "unusable audio reached the LLM"
                print(f"  fallback fired: {msg.get('fallback_reason')}")
                print(f"  coach: {msg['text']}")
                break
            if msg["type"] == "error":
                print(f"  ASR error path: {msg['message'][:88]}")
                break
        await ws.send(json.dumps({"type": "end_session"}))
        await asyncio.sleep(1)


async def main() -> None:
    health = get("/api/health")
    if health["status"] != "ok":
        print(f"backend is {health['status']}: {json.dumps(health, indent=2)}")
        sys.exit(1)
    print(f"backend ok — llm={health['llm']['provider']} asr={health['asr']['provider']} "
          f"tts={health['tts']['provider']}")

    name = f"TestKid-{uuid.uuid4().hex[:6]}"
    student = post("/api/students", {"name": name, "grade": 6})
    print(f"test student: {name}")

    await test_baseline(student)
    await test_resume(student)
    await test_fallback(student)

    await asyncio.sleep(3)
    banner("FINAL PROGRESS")
    print(json.dumps(get(f"/api/students/{student['id']}/progress"), indent=2))
    print("\nAll end-to-end checks passed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except AssertionError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        sys.exit(1)
