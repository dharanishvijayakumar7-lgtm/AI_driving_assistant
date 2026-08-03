/**
 * App.jsx — Root component
 *
 * Wires up the WebSocket hook and lays out the three main areas:
 *   1. Header bar   — branding + live connection pill
 *   2. Feed + Sidebar — annotated video | object list
 *   3. Stats bar    — FPS, lane offset, object count, connection state
 *
 * The <AlertBanner> floats absolutely inside <VideoFeed>'s container.
 */

import React from 'react'
import { useWebSocketStream }  from './hooks/useWebSocketStream'
import VideoFeed               from './components/VideoFeed'
import AlertBanner             from './components/AlertBanner'
import ObjectPanel             from './components/ObjectPanel'
import StatsBar                from './components/StatsBar'

const STATUS_LABELS = {
  connected:    'Connected',
  disconnected: 'Disconnected',
  reconnecting: 'Reconnecting',
}

export default function App() {
  const { frameSrc, metadata, connectionStatus } = useWebSocketStream()

  const objects     = metadata?.tracked_objects ?? []
  const activeAlert = metadata?.active_alert    ?? null
  const frameNumber = metadata?.frame_number    ?? null

  return (
    <div className="app" id="app-root">

      {/* ── Header ─────────────────────────────────────────── */}
      <header className="app-header" id="app-header">
        <div className="header-brand">
          <div className="header-logo" aria-hidden="true">🚗</div>
          <div>
            <p className="header-title">AI Driving Assistant</p>
            <p className="header-subtitle">Real-time CV Pipeline Dashboard</p>
          </div>
        </div>

        <div className="header-right">
          <div
            className={`conn-pill ${connectionStatus}`}
            role="status"
            aria-label={`WebSocket: ${STATUS_LABELS[connectionStatus]}`}
            id="header-conn-pill"
          >
            <div className="conn-dot" aria-hidden="true" />
            {STATUS_LABELS[connectionStatus]}
          </div>
        </div>
      </header>

      {/* ── Main feed ──────────────────────────────────────── */}
      {/* AlertBanner sits inside the VideoFeed container (position: absolute) */}
      <div style={{ position: 'relative', gridArea: 'feed', overflow: 'hidden', display: 'flex' }}>
        <VideoFeed
          frameSrc={frameSrc}
          connectionStatus={connectionStatus}
          frameNumber={frameNumber}
        />
        <AlertBanner alert={activeAlert} />
      </div>

      {/* ── Sidebar ────────────────────────────────────────── */}
      <ObjectPanel objects={objects} />

      {/* ── Stats bar ──────────────────────────────────────── */}
      <StatsBar metadata={metadata} connectionStatus={connectionStatus} />

    </div>
  )
}
