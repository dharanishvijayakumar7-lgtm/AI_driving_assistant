/**
 * StatsBar.jsx
 *
 * Bottom bar showing real-time pipeline statistics and connection state.
 *
 * Cells:
 *   1. FPS          — metadata.fps_current
 *   2. Objects      — tracked object count + DANGER/CAUTION summary
 *   3. Lane Offset  — metadata.lane_offset, rendered as a visual bar
 *   4. Connection   — "Connected" / "Disconnected" / "Reconnecting" with colour
 *
 * Props:
 *   metadata         — FrameMetadataSchema or null
 *   connectionStatus — "connected" | "disconnected" | "reconnecting"
 */

import React from 'react'

function fpsColour(fps) {
  if (fps == null || fps === 0) return 'muted'
  if (fps >= 8)  return 'green'
  if (fps >= 4)  return 'amber'
  return 'red'
}

function LaneOffsetBar({ offset }) {
  // offset is null (no lanes) or a float in [-1, 1]
  if (offset == null) {
    return (
      <span className="stat-cell-value muted" style={{ fontSize: 13 }}>
        —
      </span>
    )
  }

  // Convert [-1,1] to [0%,100%] for positioning
  const pct     = ((offset + 1) / 2) * 100            // centre = 50%
  const fromCtr = 50                                    // thumb start (centre)
  const toCtr   = pct

  const fillLeft  = Math.min(fromCtr, toCtr)
  const fillWidth = Math.abs(toCtr - fromCtr)

  // Colour: green when < 15% offset, amber < 40%, red beyond
  const absOff = Math.abs(offset)
  const barColour =
    absOff < 0.15 ? 'var(--green)'
    : absOff < 0.40 ? 'var(--amber)'
    : 'var(--red)'

  return (
    <div className="lane-bar-container" aria-label={`Lane offset: ${offset.toFixed(2)}`}>
      <div className="lane-bar-track" role="progressbar"
           aria-valuenow={offset} aria-valuemin={-1} aria-valuemax={1}>
        <div className="lane-bar-center" aria-hidden="true" />
        <div
          className="lane-bar-fill"
          aria-hidden="true"
          style={{
            left:       `${fillLeft}%`,
            width:      `${fillWidth}%`,
            background: barColour,
            opacity: 0.7,
          }}
        />
        <div
          className="lane-bar-thumb"
          aria-hidden="true"
          style={{ left: `${pct}%`, background: barColour }}
        />
      </div>
      <span style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        color: barColour,
        letterSpacing: '0.02em',
      }}>
        {offset >= 0 ? '+' : ''}{offset.toFixed(2)}
      </span>
    </div>
  )
}

function ConnectionCell({ status }) {
  const labels = {
    connected:    'Connected',
    disconnected: 'Disconnected',
    reconnecting: 'Reconnecting',
  }
  const colours = {
    connected:    'green',
    disconnected: 'red',
    reconnecting: 'amber',
  }
  return (
    <span className={`stat-cell-value ${colours[status] ?? 'muted'}`} id="stats-connection-status">
      {labels[status] ?? '—'}
    </span>
  )
}

export default function StatsBar({ metadata, connectionStatus }) {
  const fps     = metadata?.fps_current ?? null
  const offset  = metadata?.lane_offset ?? null
  const objects = metadata?.tracked_objects ?? []

  const dangerCount  = objects.filter(o => o.risk_level === 'DANGER').length
  const cautionCount = objects.filter(o => o.risk_level === 'CAUTION').length

  return (
    <footer className="stats-bar" role="contentinfo" aria-label="Pipeline statistics">

      {/* ── FPS ───────────────────────────────────────────── */}
      <div className="stat-cell" id="stats-fps-cell">
        <span className="stat-cell-icon" aria-hidden="true">⚡</span>
        <div className="stat-cell-body">
          <span className="stat-cell-label">FPS</span>
          <span className={`stat-cell-value ${fpsColour(fps)}`} id="stats-fps-value">
            {fps != null && fps > 0 ? fps.toFixed(1) : '—'}
          </span>
        </div>
      </div>

      {/* ── Objects ───────────────────────────────────────── */}
      <div className="stat-cell" id="stats-objects-cell">
        <span className="stat-cell-icon" aria-hidden="true">🎯</span>
        <div className="stat-cell-body">
          <span className="stat-cell-label">Objects</span>
          <span className="stat-cell-value" id="stats-objects-value"
                style={{ color: dangerCount > 0 ? 'var(--red)' : 'var(--cyan)' }}>
            {objects.length}
            {dangerCount > 0 && (
              <span style={{ fontSize: 10, color: 'var(--red)', marginLeft: 5 }}>
                {dangerCount}⚠
              </span>
            )}
            {cautionCount > 0 && dangerCount === 0 && (
              <span style={{ fontSize: 10, color: 'var(--amber)', marginLeft: 5 }}>
                {cautionCount}!
              </span>
            )}
          </span>
        </div>
      </div>

      {/* ── Lane Offset ───────────────────────────────────── */}
      <div className="stat-cell" id="stats-lane-cell" style={{ flex: 2 }}>
        <span className="stat-cell-icon" aria-hidden="true">🛣️</span>
        <div className="stat-cell-body" style={{ flex: 1 }}>
          <span className="stat-cell-label">Lane Offset</span>
          <LaneOffsetBar offset={offset} />
        </div>
      </div>

      {/* ── Connection ────────────────────────────────────── */}
      <div className="stat-cell" id="stats-conn-cell">
        <span className="stat-cell-icon" aria-hidden="true">📡</span>
        <div className="stat-cell-body">
          <span className="stat-cell-label">Backend</span>
          <ConnectionCell status={connectionStatus} />
        </div>
      </div>

    </footer>
  )
}
