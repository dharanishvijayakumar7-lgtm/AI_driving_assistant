/**
 * useWebSocketStream.js
 *
 * Custom hook that manages the WebSocket connection to the FastAPI backend.
 *
 * Returns:
 *   frameSrc        — data URI for the <img> tag (base64 JPEG), or null
 *   metadata        — the FrameMetadataSchema object from the last message, or null
 *   connectionStatus — "connected" | "disconnected" | "reconnecting"
 *
 * Reconnection strategy:
 *   Uses exponential backoff with jitter:
 *     attempt 1 → 1 s, attempt 2 → 2 s, attempt 3 → 4 s, … capped at 30 s.
 *   The reconnect timer is reset on clean close (stream ended) and on
 *   component unmount.
 */

import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = 'ws://localhost:8000/ws/stream'
const MAX_BACKOFF_MS = 30_000
const BASE_BACKOFF_MS = 1_000

export function useWebSocketStream() {
  const [frameSrc, setFrameSrc]               = useState(null)
  const [metadata, setMetadata]               = useState(null)
  const [connectionStatus, setConnectionStatus] = useState('disconnected')

  // Refs so the reconnect callback always sees current values
  const wsRef           = useRef(null)
  const attemptRef      = useRef(0)
  const timeoutRef      = useRef(null)
  const mountedRef      = useRef(true)

  const clearTimer = useCallback(() => {
    if (timeoutRef.current !== null) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
  }, [])

  const connect = useCallback(() => {
    if (!mountedRef.current) return

    // Close any lingering socket
    if (wsRef.current) {
      wsRef.current.onclose = null   // suppress reconnect on manual close
      wsRef.current.close()
      wsRef.current = null
    }

    setConnectionStatus('reconnecting')

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) { ws.close(); return }
      attemptRef.current = 0
      setConnectionStatus('connected')
    }

    ws.onmessage = (event) => {
      if (!mountedRef.current) return
      try {
        const payload = JSON.parse(event.data)
        // frame_b64 may or may not carry the data: prefix — normalise it
        if (payload.frame_b64) {
          setFrameSrc(`data:image/jpeg;base64,${payload.frame_b64}`)
        }
        if (payload.metadata) {
          setMetadata(payload.metadata)
        }
      } catch (err) {
        console.warn('[useWebSocketStream] Failed to parse message:', err)
      }
    }

    ws.onerror = (err) => {
      // onerror is always followed by onclose — let onclose handle retry
      console.warn('[useWebSocketStream] WebSocket error', err)
    }

    ws.onclose = (event) => {
      if (!mountedRef.current) return
      setConnectionStatus('disconnected')
      wsRef.current = null

      // Don't reconnect on a clean close from the server (code 1000 = stream ended)
      // but do reconnect on abnormal closes (network drop, server crash, etc.)
      if (event.code === 1000) {
        console.info('[useWebSocketStream] Stream ended cleanly (video finished).')
        return
      }

      // Exponential backoff with ±20% jitter
      const delay = Math.min(
        BASE_BACKOFF_MS * Math.pow(2, attemptRef.current),
        MAX_BACKOFF_MS
      )
      const jitter = delay * 0.2 * (Math.random() - 0.5)
      attemptRef.current += 1

      console.info(
        `[useWebSocketStream] Reconnecting in ${Math.round(delay + jitter)} ms ` +
        `(attempt ${attemptRef.current})`
      )
      timeoutRef.current = setTimeout(connect, delay + jitter)
    }
  }, [clearTimer])

  useEffect(() => {
    mountedRef.current = true
    connect()

    return () => {
      mountedRef.current = false
      clearTimer()
      if (wsRef.current) {
        wsRef.current.onclose = null   // prevent reconnect on unmount
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [connect, clearTimer])

  return { frameSrc, metadata, connectionStatus }
}
