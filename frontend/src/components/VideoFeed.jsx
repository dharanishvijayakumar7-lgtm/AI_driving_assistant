/**
 * VideoFeed.jsx
 *
 * Renders the annotated frame coming from the backend WebSocket.
 * The frame is fully annotated server-side (bounding boxes, lane overlay,
 * depth panel, alert banner baked in). This component simply displays it
 * and overlays the connection-state UX on top.
 *
 * Props:
 *   frameSrc         — data URI (base64 JPEG), or null when disconnected
 *   connectionStatus — "connected" | "disconnected" | "reconnecting"
 *   frameNumber      — monotonically increasing frame counter from metadata
 */

import React from 'react'

export default function VideoFeed({ frameSrc, connectionStatus, frameNumber }) {
  const isConnected    = connectionStatus === 'connected'
  const isReconnecting = connectionStatus === 'reconnecting'

  return (
    <div className="feed-area" id="video-feed-container" role="img" aria-label="Live driving feed">
      {/* ── Live frame ──────────────────────────────────────── */}
      {frameSrc ? (
        <>
          <img
            id="video-frame"
            className="video-frame"
            src={frameSrc}
            alt="Annotated driving feed"
            draggable={false}
          />
          {/* Frame counter watermark */}
          {frameNumber != null && (
            <div className="frame-badge" aria-hidden="true">
              #{frameNumber}
            </div>
          )}
        </>
      ) : (
        /* Placeholder when no frame has been received yet */
        <div className="feed-placeholder" aria-live="polite">
          <div className="feed-placeholder-icon">🎥</div>
          <p className="feed-placeholder-title">Waiting for stream…</p>
          <p className="feed-placeholder-sub">
            Start the FastAPI backend, then the feed will appear here automatically.
          </p>
          <code className="feed-placeholder-code">python src/run_server.py</code>
        </div>
      )}

      {/* ── Reconnecting overlay (shown on top of stale frame) ── */}
      {isReconnecting && (
        <div className="feed-reconnecting" aria-live="assertive">
          <div className="spinner" aria-hidden="true" />
          <p>Reconnecting…</p>
        </div>
      )}

      {/* ── Disconnected overlay (only when no frame cached) ──── */}
      {!isConnected && !isReconnecting && !frameSrc && (
        <div
          style={{
            position: 'absolute', bottom: 16, left: '50%',
            transform: 'translateX(-50%)',
            display: 'flex', alignItems: 'center', gap: 8,
            fontSize: 12,
            color: 'var(--red)',
            background: 'rgba(239,68,68,0.12)',
            border: '1px solid rgba(239,68,68,0.3)',
            padding: '6px 14px',
            borderRadius: 20,
          }}
          role="alert"
        >
          <span>⚠</span>
          Backend disconnected — ensure the server is running on port 8000
        </div>
      )}
    </div>
  )
}
