import { useEffect, useRef } from 'react'

/**
 * The conversation itself: what the coach said, what we heard the student say,
 * and playback of the coach's voice.
 *
 * The ASR confidence is shown next to each student line on purpose. When the
 * coach answers something odd, the transcript is almost always why — and that
 * is exactly what you want visible while demoing.
 */

function AudioPlayer({ audio, mime }) {
  const ref = useRef(null)

  useEffect(() => {
    if (!audio || !ref.current) return
    // Autoplay is allowed here: the student clicked to start the session, so
    // the page already has an activation gesture.
    ref.current.play().catch(() => {})
  }, [audio])

  if (!audio) return null
  return <audio ref={ref} controls src={`data:${mime};base64,${audio}`} />
}

function confidenceClass(c) {
  if (c === null || c === undefined) return ''
  if (c >= 0.75) return 'conf-good'
  if (c >= 0.55) return 'conf-ok'
  return 'conf-low'
}

export default function SessionView({ messages, status }) {
  const endRef = useRef(null)
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, status])

  return (
    <div className="session">
      {messages.map((m, i) => (
        <div key={i} className={`turn turn-${m.role}`}>
          <div className="turn-who">{m.role === 'coach' ? 'Coach' : 'You'}</div>
          <div className="turn-text">
            {m.text}
            {m.fallback && <span className="tag tag-warn">didn&apos;t hear you</span>}
            {m.safety_action && m.safety_action !== 'pass' && (
              <span className="tag tag-safety">safety: {m.safety_action}</span>
            )}
            {m.role === 'student' && m.confidence != null && (
              <span className={`tag ${confidenceClass(m.confidence)}`}>
                heard {(m.confidence * 100).toFixed(0)}%
              </span>
            )}
          </div>
          {m.audio && <AudioPlayer audio={m.audio} mime={m.mime} />}
        </div>
      ))}
      {status && (
        <div className="turn turn-status">
          <span className="dots">
            <i /> <i /> <i />
          </span>
          {status}
        </div>
      )}
      <div ref={endRef} />
    </div>
  )
}
