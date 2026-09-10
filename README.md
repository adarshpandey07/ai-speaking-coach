# AI Speaking Confidence Coach — Local MVP

A voice-based English speaking coach for grade 6-7 students (CBSE, tier-2/3
India). The student opens a browser, presses one button, and has a real spoken
conversation with an AI tutor that follows a weekly speaking curriculum.

**Phase 1 is speaking and personality confidence only.** There is no
academic-subject content, and the coach is instructed to refuse it.

Everything runs on one laptop. No Kubernetes, no cloud services, no managed
queue — Postgres is the only thing you have to start.

---

## What actually happens in a session

```
browser mic ──WebSocket──▶ FastAPI
                             │
                             ├─▶ ASR  (faster-whisper)  → transcript + confidence
                             │        │
                             │        └─ low confidence / silence?
                             │           └─▶ fallback handler → graceful re-prompt
                             │               (the LLM is never called)
                             │
                             ├─▶ LangGraph agent (Postgres checkpointer)
                             │        └─▶ LLM SDK call  → draft reply
                             │
                             ├─▶ safety filter → pass / sanitize / regenerate / canned
                             │        (nothing reaches TTS without passing this)
                             │
                             └─▶ TTS → audio ──▶ back to the browser
```

At the end of the session a FastAPI background task scores the student's speech
and updates the rolling mastery average. Nothing blocks the conversation.

Two design points worth knowing before you read the code:

- **The LangGraph Postgres checkpointer is load-bearing.** Session state lives
  in Postgres keyed by session id. If wifi drops mid-conversation, reopening
  the same session id resumes the conversation instead of restarting it. The
  frontend does this automatically, with backoff.
- **The safety filter is a gate, not a suggestion.** It sits inside the graph
  between the LLM and TTS. A blocked reply is regenerated once; if the rewrite
  also fails, a bland canned line is spoken. An unchecked reply is never sent
  to a child.

---

## Setup

Prerequisites: **Python 3.11+** (3.12 recommended), **Node 18+**, and either
Docker or a local Postgres.

### 1. Postgres

Either run it in Docker:

```bash
docker run -d --name speaking-coach-pg \
  -e POSTGRES_USER=coach -e POSTGRES_PASSWORD=coach -e POSTGRES_DB=speaking_coach \
  -p 5432:5432 postgres:16-alpine
```

…or use a native install:

```bash
brew install postgresql@16 && brew services start postgresql@16
createdb speaking_coach
createuser coach --pwprompt        # use: coach
psql -d speaking_coach -c "GRANT ALL ON DATABASE speaking_coach TO coach;"
```

SQLite is deliberately not supported — the LangGraph checkpointer is the
Postgres one.

**Tables are created automatically** on first backend start
(`Base.metadata.create_all`), including the LangGraph checkpoint tables. There
is no separate migration step for the MVP.

### 2. Backend

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env      # defaults work as-is, see "Configuration" below

uvicorn app.main:app --reload --port 8010
```

First start downloads the Whisper model (~75 MB for `base.en`) into the
HuggingFace cache. It is loaded in the background at startup, so the first
student turn is not the one that waits for it.

Check it came up clean:

```bash
curl -s localhost:8010/api/health | python -m json.tool
```

`"status": "ok"` means Postgres is reachable and the LLM is configured.
`"degraded"` tells you which one is not, and why.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**, type a name, press **Start Session**, and allow
the microphone when the browser asks.

> Use `localhost`, not a LAN IP. `getUserMedia` only works on a secure context,
> and `http://localhost` counts as one while `http://192.168.x.x` does not.

The Vite dev server proxies `/api` and `/ws` to port 8010, so the browser only
ever sees one origin.

---

## Verifying it works

`scripts/e2e_check.py` drives the whole stack the way the browser does: it
synthesises a child's voice, streams it over the WebSocket in chunks, and
asserts on what comes back. It covers the three things that are easy to break
and impossible to eyeball.

```bash
# with the backend already running on 8010
python scripts/e2e_check.py
```

| Check | Asserts |
| --- | --- |
| Day-one baseline | A student with no baseline gets a baseline session, and a `BaselineAssessment` row is persisted afterwards |
| Reconnect-and-resume | After an abrupt socket close mid-session, reopening the same session id returns `resumed: true` at the same turn index, and the next turn continues rather than restarting |
| Fallback handler | Unusable audio produces `fallback: true` and never reaches the LLM |

It exits non-zero on the first failed assertion, and creates a throwaway
`TestKid-xxxxxx` student each run.

### Bringing the stack back up

The three pieces are independent. If you have stopped them:

```bash
docker start speaking-coach-pg              # data persists across stop/start
uvicorn app.main:app --reload --port 8010   # from the repo root, venv active
cd frontend && npm run dev                  # http://localhost:5173
```

The backend keeps ~300 MB resident once the Whisper model is loaded. On a
machine that is already deep into swap the OS may kill it at startup — free
memory first rather than retrying.

---

## Configuration

All of it lives in `.env` (see `.env.example`). One provider, one model — there
is no routing or fallback logic between LLMs.

| Variable | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` | `claude_cli` | `claude_cli` shells out to the local `claude` binary — no API key, no metered spend. `anthropic` calls the SDK directly. |
| `LLM_MODEL` | `claude-sonnet-5` | |
| `LLM_API_KEY` | *(empty)* | Required only for `LLM_PROVIDER=anthropic`. |
| `ASR_PROVIDER` | `faster_whisper` | Local, offline. |
| `ASR_MODEL_SIZE` | `base.en` | `small.en` is more accurate and slower. |
| `ASR_MIN_CONFIDENCE` | `0.55` | Below this the fallback handler answers instead of the LLM. |
| `TTS_PROVIDER` | `edge_tts` | Free neural voices, needs network, no API key. |
| `TTS_VOICE` | `en-IN-NeerjaNeural` | |
| `TTS_OFFLINE_FALLBACK` | `true` | Falls back to the macOS `say` binary if the primary TTS fails. |
| `DATABASE_URL` | `postgresql://coach:coach@localhost:5432/speaking_coach` | |
| `SESSION_MAX_TURNS` | `10` | ~10 minutes of conversation. |
| `BASELINE_TURNS` | `3` | Length of the day-one assessment. |

### Swapping a provider

`app/voice/asr.py`, `app/voice/tts.py` and `app/agent/llm.py` each define an
abstract base class and select an implementation from config. To add a hosted
recogniser, implement `ASRProvider.transcribe` (returning transcript **and**
confidence — the fallback handler depends on it) and add one branch to
`get_asr()`.

---

## The curriculum

Weeks 1-4 live in `app/curriculum/weeks.py` and are seeded into Postgres on
startup. The week drives the coach's system prompt, its difficulty, and the
target answer length used for scoring.

| Week | Topic | Target words/turn |
| --- | --- | --- |
| 1 | Talking about myself | 12 |
| 2 | Describing things and places | 22 |
| 3 | Telling a story | 32 |
| 4 | Opinions and reasons | 40 |

A student advances one day per completed session, and one week per five days.

### Baseline assessment

A student with no `BaselineAssessment` row gets a **baseline session** instead
of a curriculum session: three fixed prompts, no teaching, no correction. It is
scored with exactly the same rules a regular session uses — if it were not,
"improvement since baseline" would be comparing two different rulers.

### Mastery scoring

Rule-based, in `app/mastery/tracker.py`. Four sub-scores, all 0-100:

- **fluency** — words per minute, penalised by hesitation markers
- **vocabulary** — type/token ratio and content-word share
- **elaboration** — words per turn against the week's target
- **clarity** — mean ASR confidence

`overall` is their weighted sum, and `rolling_average` weights the last five
sessions 5·4·3·2·1, newest first. This is a placeholder for a real
pronunciation-assessment API; the shape of the output is designed so that
swapping the internals does not change anything downstream.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Postgres + LLM readiness, and the active providers |
| `POST` | `/api/students` | Get-or-create a student by name |
| `GET` | `/api/students/{id}/progress` | Week, streak, baseline, mastery |
| `POST` | `/api/students/{id}/sessions` | Start a session (baseline or regular) |
| `GET` | `/api/sessions/{id}/transcript` | Stored turns |
| `POST` | `/api/sessions/{id}/complete` | Score a session out of band |
| `WS` | `/ws/session/{session_id}` | The live conversation |

### WebSocket protocol

Client → server: binary audio chunks, then a JSON control frame.

```jsonc
{"type": "end_utterance"}     // done speaking, transcribe and reply
{"type": "cancel_utterance"}  // discard the buffered audio
{"type": "end_session"}       // stop early; scoring runs in the background
{"type": "ping"}              // keepalive
```

Server → client, all JSON:

```jsonc
{"type": "ready",      "resumed": false, "transcript": [], "coach": {"text": "...", "audio": "<base64>", "mime": "audio/mpeg"}}
{"type": "thinking",   "stage": "listening" | "coach"}
{"type": "transcript", "text": "...", "confidence": 0.81}
{"type": "coach",      "text": "...", "audio": "<base64>", "fallback": false, "safety_action": "pass", "turn": 3, "finished": false}
{"type": "session_end","reason": "turns_complete"}
{"type": "error",      "code": "asr_failed", "message": "..."}
```

---

## Error handling

Each of these is a real, exercised path, not a `try/except: pass`:

| Failure | What happens |
| --- | --- |
| Mic permission denied | Named message telling the student how to re-enable it (`Recorder.jsx` distinguishes `NotAllowedError`, `NotFoundError`, `NotReadableError`) |
| Missing LLM API key | Logged loudly at startup, reported by `/api/health`, and the WebSocket closes with an explanatory message rather than failing mid-turn |
| WebSocket disconnect | Client reconnects with exponential backoff, reopens the same session id, and the LangGraph checkpoint resumes the conversation |
| Low-confidence ASR / silence | Fallback handler answers with an escalating re-prompt; the LLM is not called. After 3 misses in a row it suggests checking the mic and ends cleanly |
| Unsafe LLM output | Safety filter blocks it, the graph regenerates once, and falls back to a canned safe line if the rewrite also fails |
| TTS failure | Falls back to the offline macOS voice; if that also fails the turn still arrives as text |
| Postgres unreachable | Startup fails immediately with the exact `docker run` command to fix it |

Not production-grade — but nothing here is silently broken.

---

## Project layout

```
app/
  main.py              FastAPI app, WebSocket session loop
  config.py            All env vars
  db.py                Postgres engine/session (SQLAlchemy)
  models.py            Student, Session, Turn, BaselineAssessment,
                       MasteryScore, AdaptationDirective, CurriculumWeek
  store.py             Synchronous data access, kept out of main.py
  agent/
    graph.py           LangGraph state graph + Postgres checkpointer
    llm.py             Single LLM provider, called through its own SDK
    prompts.py         System prompts per curriculum week
    safety.py          Age-appropriate output filter
    fallback.py        Low-confidence ASR / silence handling
  voice/
    asr.py             ASRProvider ABC + faster-whisper
    tts.py             TTSProvider ABC + edge-tts and macOS `say`
  mastery/
    tracker.py         Rule-based scoring + rolling 5-session average
    baseline.py        Day-one assessment
  curriculum/
    weeks.py           Weeks 1-4
frontend/
  src/
    App.jsx
    components/{Recorder,SessionView,ProgressBadge}.jsx
    ws/client.js       WebSocket client with reconnect-and-resume
```

`store.py` and `agent/llm.py` are the two files added beyond the original
structure: the first keeps SQLAlchemy out of the WebSocket handler, the second
keeps the provider SDK out of the graph.

---

## Deliberately out of scope

- No LiteLLM, no multi-model routing
- No Go service, no Kubernetes, no Docker beyond optionally running Postgres
- No parent/teacher dashboard UI. The `adaptation_directives` table exists and
  is wired into the system prompt; a seeded example row is inserted with
  `active = false`. Flip it to see it take effect:
  ```sql
  UPDATE adaptation_directives SET active = true WHERE student_id = '<id>';
  ```
- No production auth — a student is identified by the name they type
- No real pronunciation-assessment API; scoring is rule-based

---

## Troubleshooting

**`Cannot reach Postgres…`** — the container is not running. The error message
contains the exact `docker run` to fix it.

**First turn takes ~30s** — the Whisper model is still downloading. Watch the
backend log for `ASR model warm`.

**Coach replies but there is no audio** — `edge-tts` needs network. Either
reconnect, or set `TTS_PROVIDER=macos_say` for a fully offline voice.

**`heard 30%` tags on your speech** — the room is noisy or the mic is far away.
Lower `ASR_MIN_CONFIDENCE` for a demo, or switch `ASR_MODEL_SIZE=small.en` for
better accuracy at roughly 2× the latency.

**Mic button does nothing** — you are on a LAN IP. Use `http://localhost:5173`.
