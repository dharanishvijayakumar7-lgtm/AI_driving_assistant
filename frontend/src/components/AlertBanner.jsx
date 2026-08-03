/**
 * AlertBanner.jsx
 *
 * Renders a prominent, React-driven alert banner using metadata.active_alert.
 *
 * Design intent:
 *   The backend already bakes an alert banner into the annotated frame image.
 *   This SEPARATE React banner proves the frontend is actually consuming and
 *   acting on the structured metadata, not just displaying a dumb video feed.
 *   It sits as a floating pill above the video frame.
 *
 * Props:
 *   alert — AlertSchema object or null:
 *     { active, message, severity, track_id, duration_seconds }
 */

import React from 'react'

export default function AlertBanner({ alert }) {
  if (!alert || !alert.active) return null

  const isDanger  = alert.severity === 'DANGER'
  const icon      = isDanger ? '🚨' : '⚠️'
  const className = `alert-banner ${isDanger ? 'danger' : 'caution'}`

  // Clean up the message — backend sometimes prefixes with '⚠ '
  const cleanMessage = alert.message
    .replace(/^[⚠🚨]\s*/u, '')
    .replace('COLLISION RISK — ', '')

  return (
    <div className={className} role="alert" aria-live="assertive" id="alert-banner">
      <span className="alert-icon" aria-hidden="true">{icon}</span>
      <span className="alert-text">
        {isDanger ? '⚡ COLLISION RISK' : '⚠ CAUTION'} — {cleanMessage}
      </span>
      {alert.duration_seconds > 0 && (
        <span className="alert-duration">
          ({alert.duration_seconds.toFixed(1)}s)
        </span>
      )}
    </div>
  )
}
