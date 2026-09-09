import { useCallback, useEffect, useRef, useState } from 'react'
import ProgressBadge from './components/ProgressBadge'
import Recorder from './components/Recorder'
import SessionView from './components/SessionView'
import { SessionSocket } from './ws/client'

const STORAGE_KEY = 'speaking-coach.session'

// No production auth in the MVP — a name is the whole identity.
const DEFAULT_NAME = 'Riya'

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

export default function App() {
  const [phase, setPhase] = useState('idle') // idle | connecting | live | ended
  const [name, setName] = useState(DEFAULT_NAME)
  const [messages, setMessages] = useState([])
  const [meta, setMeta] = useState(null)
  const [status, setStatus] = useState('')
  const [connection, setConnection] = useState('')
  const [error, setError] = useState('')
  const [coachTurn, setCoachTurn] = useState(false)
  const [summary, setSummary] = useState(null)
  const [resumable, setResumable] = useState(null)
  const socketRef = useRef(null)

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved) {
      try {
        setResumable(JSON.parse(saved))
      } catch {
        localStorage.removeItem(STORAGE_KEY)
      }
    }
  }, [])

  const handleMessage = useCallback((msg) => {
    switch (msg.type) {
      case 'ready':
        setMeta(msg)
        setStatus('')
        setPhase('live')
        setMessages(() => {
          const prior = (msg.transcript || [])
            .filter((t) => t.index > 0)
            .map((t) => ({
              role: t.role === 'coach' ? 'coach' : 'student',
              text: t.text,
              confidence: t.confidence,
              safety_action: t.safety_action,
            }))
          return [
            ...prior,
            msg.coach && {
              role: 'coach',
              text: msg.coach.text,
              audio: msg.coach.audio,
              mime: msg.coach.mime,
            },
          ].filter(Boolean)
        })
        setCoachTurn(false)
        break

      case 'thinking':
        setStatus(msg.stage === 'coach' ? 'Coach is thinking…' : 'Listening to you…')
        setCoachTurn(true)
        break

      case 'transcript':
        setMessages((m) => [
          ...m,
          { role: 'student', text: msg.text || '(nothing heard)', confidence: msg.confidence },
        ])
        break

      case 'coach':
        setStatus('')
        setCoachTurn(false)
        setMeta((prev) => (prev ? { ...prev, turn: msg.turn } : prev))
        setMessages((m) => [
          ...m,
          {
            role: 'coach',
            text: msg.text,
            audio: msg.audio,
            mime: msg.mime,
            fallback: msg.fallback,
            safety_action: msg.safety_action,
          },
        ])
        break

      case 'session_end':
        setPhase('ended')
        setStatus('')
        localStorage.removeItem(STORAGE_KEY)
        socketRef.current?.close()
        break

      case 'error':
        setError(msg.message || 'Something went wrong.')
        setCoachTurn(false)
        setStatus('')
        break

      default:
        break
    }
  }, [])

  const handleStatus = useCallback((state, info) => {
    switch (state) {
      case 'connected':
        setConnection('')
        break
      case 'reconnected':
        setConnection('')
        setError('')
        break
      case 'reconnecting':
        setConnection(`Connection lost — reconnecting (try ${info.attempt})…`)
        break
      case 'failed':
        setConnection('')
        setError(
          'Could not reconnect. Your session is saved — reload the page and press Resume to carry on.',
        )
        break
      default:
        break
    }
  }, [])

  const openSocket = useCallback(
    (sessionId) => {
      const socket = new SessionSocket(sessionId, {
        onMessage: handleMessage,
        onStatus: handleStatus,
      })
      socketRef.current = socket
      socket.connect()
    },
    [handleMessage, handleStatus],
  )

  const startSession = useCallback(async () => {
    setError('')
    setSummary(null)
    setMessages([])
    setPhase('connecting')
    try {
      const student = await postJSON('/api/students', { name: name.trim() || DEFAULT_NAME })
      const session = await postJSON(`/api/students/${student.id}/sessions`)
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ sessionId: session.session_id, studentId: student.id, name: student.name }),
      )
      setResumable(null)
      openSocket(session.session_id)
    } catch (err) {
      setPhase('idle')
      setError(
        `Could not start a session: ${err.message}. Is the backend running on port 8010?`,
      )
    }
  }, [name, openSocket])

  const resumeSession = useCallback(() => {
    if (!resumable) return
    setError('')
    setMessages([])
    setPhase('connecting')
    openSocket(resumable.sessionId)
  }, [resumable, openSocket])

  const endSession = useCallback(async () => {
    socketRef.current?.endSession()
    setPhase('ended')
  }, [])

  useEffect(() => {
    if (phase !== 'ended') return
    const saved = resumable || JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null')
    const studentId = saved?.studentId
    if (!studentId) return
    // The score is written by a background task, so give it a moment.
    const timer = setTimeout(async () => {
      try {
        const res = await fetch(`/api/students/${studentId}/progress`)
        if (res.ok) setSummary(await res.json())
      } catch {
        /* summary is a nice-to-have */
      }
    }, 1500)
    return () => clearTimeout(timer)
  }, [phase, resumable])

  return (
    <div className="app">
      <header>
        <h1>Speaking Coach</h1>
        {meta && (
          <ProgressBadge
            week={meta.week_number}
            day={meta.current_day}
            streak={meta.streak_days}
            topic={meta.week_topic}
            mode={meta.mode}
          />
        )}
      </header>

      {connection && <div className="banner banner-warn">{connection}</div>}
      {error && <div className="banner banner-error">{error}</div>}

      {phase === 'idle' && (
        <div className="card start-card">
          <h2>Ready to practise speaking?</h2>
          <p className="muted">
            We&apos;ll chat out loud for about ten minutes. There are no marks and no
            wrong answers.
          </p>
          <label htmlFor="name">Your name</label>
          <input
            id="name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Your name"
          />
          <button className="primary" onClick={startSession}>
            Start Session
          </button>
          {resumable && (
            <button className="ghost" onClick={resumeSession}>
              Resume {resumable.name}&apos;s last session
            </button>
          )}
        </div>
      )}

      {phase === 'connecting' && <div className="card muted">Connecting…</div>}

      {(phase === 'live' || phase === 'ended') && (
        <>
          <SessionView messages={messages} status={status} />
          {phase === 'live' && (
            <footer>
              <Recorder
                socket={socketRef.current}
                disabled={coachTurn}
                onError={setError}
              />
              <div className="footer-meta">
                <span>
                  Turn {meta?.turn ?? 0} / {meta?.max_turns ?? '-'}
                </span>
                <button className="ghost small" onClick={endSession}>
                  End session
                </button>
              </div>
            </footer>
          )}
        </>
      )}

      {phase === 'ended' && (
        <div className="card end-card">
          <h2>Nice work today! 🎉</h2>
          {summary ? (
            <div className="scores">
              {summary.baseline && (
                <div className="score">
                  <span className="score-value">{summary.baseline.overall.toFixed(0)}</span>
                  <span className="score-label">baseline</span>
                </div>
              )}
              {summary.mastery && (
                <>
                  <div className="score">
                    <span className="score-value">{summary.mastery.overall.toFixed(0)}</span>
                    <span className="score-label">this session</span>
                  </div>
                  <div className="score">
                    <span className="score-value">
                      {summary.mastery.rolling_average.toFixed(0)}
                    </span>
                    <span className="score-label">rolling avg</span>
                  </div>
                </>
              )}
              <div className="score">
                <span className="score-value">🔥 {summary.streak_days}</span>
                <span className="score-label">day streak</span>
              </div>
            </div>
          ) : (
            <p className="muted">Scoring your session…</p>
          )}
          <button className="primary" onClick={() => setPhase('idle')}>
            Back to start
          </button>
        </div>
      )}
    </div>
  )
}
