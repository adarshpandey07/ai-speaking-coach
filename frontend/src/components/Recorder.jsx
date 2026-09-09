import { useCallback, useEffect, useRef, useState } from 'react'

/**
 * Microphone capture. Chunks are streamed to the backend as they are produced
 * rather than uploaded in one blob at the end, so the server has most of the
 * audio buffered by the time the student stops talking.
 *
 * Every getUserMedia failure a real classroom produces is named explicitly —
 * "recording failed" tells a 12-year-old nothing.
 */

const MIME_CANDIDATES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/mp4',
]

function pickMimeType() {
  if (typeof MediaRecorder === 'undefined') return null
  return MIME_CANDIDATES.find((m) => MediaRecorder.isTypeSupported(m)) ?? null
}

function describeMicError(err) {
  switch (err?.name) {
    case 'NotAllowedError':
    case 'SecurityError':
      return 'Microphone access was blocked. Click the padlock in the address bar, allow the microphone, then reload.'
    case 'NotFoundError':
    case 'OverconstrainedError':
      return 'No microphone was found. Plug one in or check your system sound settings.'
    case 'NotReadableError':
      return 'Your microphone is busy in another app. Close it and try again.'
    default:
      return `Could not start the microphone: ${err?.message || err}`
  }
}

export default function Recorder({ socket, disabled, onError, onRecordingChange }) {
  const [recording, setRecording] = useState(false)
  const [supported] = useState(() => pickMimeType())
  const streamRef = useRef(null)
  const recorderRef = useRef(null)

  useEffect(() => onRecordingChange?.(recording), [recording, onRecordingChange])

  useEffect(() => {
    return () => {
      recorderRef.current?.state === 'recording' && recorderRef.current.stop()
      streamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  const start = useCallback(async () => {
    if (!supported) {
      onError('This browser cannot record audio. Try Chrome, Edge or Safari.')
      return
    }
    try {
      if (!streamRef.current) {
        streamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true },
        })
      }
      const recorder = new MediaRecorder(streamRef.current, { mimeType: supported })
      recorder.ondataavailable = async (event) => {
        if (event.data && event.data.size > 0) {
          socket.sendAudioChunk(await event.data.arrayBuffer())
        }
      }
      recorder.onstop = () => socket.endUtterance()
      recorderRef.current = recorder
      recorder.start(250) // stream a chunk every 250 ms
      setRecording(true)
    } catch (err) {
      onError(describeMicError(err))
    }
  }, [socket, supported, onError])

  const stop = useCallback(() => {
    if (recorderRef.current?.state === 'recording') recorderRef.current.stop()
    setRecording(false)
  }, [])

  return (
    <div className="recorder">
      <button
        className={`mic ${recording ? 'recording' : ''}`}
        onClick={recording ? stop : start}
        disabled={disabled}
        aria-label={recording ? 'Stop speaking' : 'Start speaking'}
      >
        <span className="mic-icon">{recording ? '■' : '🎤'}</span>
      </button>
      <p className="mic-label">
        {disabled
          ? 'Wait for the coach…'
          : recording
            ? 'Listening — tap when you finish'
            : 'Tap and speak your answer'}
      </p>
    </div>
  )
}
