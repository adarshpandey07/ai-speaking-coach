/**
 * WebSocket client for a live coaching session.
 *
 * The reconnect logic is the important part. A dropped socket is not a lost
 * session: the backend keeps the conversation in a LangGraph Postgres
 * checkpoint keyed by session id, so reopening the same id resumes the
 * conversation. This client therefore retries with backoff and reuses the
 * session id rather than starting a new session.
 */

const MAX_RETRIES = 6
const BASE_DELAY_MS = 800

export class SessionSocket {
  constructor(sessionId, handlers = {}) {
    this.sessionId = sessionId
    this.handlers = handlers
    this.ws = null
    this.retries = 0
    this.closedByUser = false
    this.pingTimer = null
  }

  get isOpen() {
    return this.ws && this.ws.readyState === WebSocket.OPEN
  }

  connect() {
    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${proto}://${window.location.host}/ws/session/${this.sessionId}`
    const ws = new WebSocket(url)
    ws.binaryType = 'arraybuffer'
    this.ws = ws

    ws.onopen = () => {
      const wasRetry = this.retries > 0
      this.retries = 0
      this.handlers.onStatus?.(wasRetry ? 'reconnected' : 'connected')
      // A keepalive ping stops idle proxies from culling the socket while a
      // student is thinking about their answer.
      this.pingTimer = setInterval(() => {
        if (this.isOpen) this.ws.send(JSON.stringify({ type: 'ping' }))
      }, 20000)
    }

    ws.onmessage = (event) => {
      let msg
      try {
        msg = JSON.parse(event.data)
      } catch {
        return
      }
      if (msg.type === 'pong') return
      this.handlers.onMessage?.(msg)
    }

    ws.onerror = () => this.handlers.onStatus?.('error')

    ws.onclose = () => {
      clearInterval(this.pingTimer)
      if (this.closedByUser) {
        this.handlers.onStatus?.('closed')
        return
      }
      if (this.retries >= MAX_RETRIES) {
        this.handlers.onStatus?.('failed')
        return
      }
      const delay = BASE_DELAY_MS * 2 ** this.retries
      this.retries += 1
      this.handlers.onStatus?.('reconnecting', { attempt: this.retries, delay })
      setTimeout(() => this.connect(), delay)
    }
  }

  sendAudioChunk(arrayBuffer) {
    if (this.isOpen) this.ws.send(arrayBuffer)
  }

  send(type, payload = {}) {
    if (this.isOpen) this.ws.send(JSON.stringify({ type, ...payload }))
  }

  endUtterance() {
    this.send('end_utterance')
  }

  endSession() {
    this.send('end_session')
  }

  close() {
    this.closedByUser = true
    clearInterval(this.pingTimer)
    this.ws?.close()
  }
}
