/**
 * ObjectPanel.jsx
 *
 * Sidebar panel listing all tracked objects from metadata.tracked_objects.
 * Each card shows: track ID, class, distance, closing speed, TTC, risk level.
 * Cards are colour-coded to match the risk level system used in the backend.
 *
 * Props:
 *   objects — array of TrackedObjectSchema (or empty array):
 *     { track_id, class_name, bbox, confidence, estimated_distance_m,
 *       closing_speed_mps, ttc_seconds, risk_level, in_ego_lane }
 */

import React from 'react'

/** Map COCO class names to emoji icons for quick visual parsing. */
const CLASS_ICONS = {
  car:        '🚗',
  truck:      '🚛',
  bus:        '🚌',
  motorcycle: '🏍',
  bicycle:    '🚲',
  person:     '🚶',
}

function classIcon(name) {
  return CLASS_ICONS[name?.toLowerCase()] ?? '📦'
}

/** Sort priority: DANGER first, then CAUTION, then SAFE. */
const RISK_ORDER = { DANGER: 0, CAUTION: 1, SAFE: 2 }

function riskClass(riskLevel) {
  return (riskLevel ?? 'SAFE').toLowerCase()
}

function formatDistance(d) {
  if (d == null) return '—'
  return `${d.toFixed(1)} m`
}

function formatSpeed(s) {
  if (s == null || s <= 0) return '—'
  return `${s.toFixed(1)} m/s`
}

function formatTTC(t) {
  if (t == null) return '—'
  if (t > 99) return '>99 s'
  return `${t.toFixed(1)} s`
}

export default function ObjectPanel({ objects = [] }) {
  // Sort: highest-risk first, then by track_id for stability
  const sorted = [...objects].sort((a, b) => {
    const riskDiff =
      (RISK_ORDER[a.risk_level] ?? 2) - (RISK_ORDER[b.risk_level] ?? 2)
    return riskDiff !== 0 ? riskDiff : a.track_id - b.track_id
  })

  return (
    <div className="sidebar">
      <div className="panel-header">
        <span className="panel-title">Tracked Objects</span>
        {objects.length > 0 && (
          <span className="panel-count" aria-label={`${objects.length} objects`}>
            {objects.length}
          </span>
        )}
      </div>

      <div className="object-list" role="list" id="object-panel-list">
        {sorted.length === 0 ? (
          <div className="object-empty">
            <div className="object-empty-icon">📡</div>
            <p className="object-empty-text">No objects detected</p>
          </div>
        ) : (
          sorted.map((obj) => {
            const risk = (obj.risk_level ?? 'SAFE').toUpperCase()
            return (
              <article
                key={obj.track_id}
                className={`object-card ${riskClass(risk)}`}
                role="listitem"
                id={`object-card-${obj.track_id}`}
                aria-label={`${obj.class_name} #${obj.track_id}, risk: ${risk}`}
              >
                <div className="object-card-header">
                  <div className="object-identity">
                    <span className="object-class-icon" aria-hidden="true">
                      {classIcon(obj.class_name)}
                    </span>
                    <span className="object-label">
                      {obj.class_name}
                    </span>
                    <span className="object-id">#{obj.track_id}</span>
                  </div>
                  <span className={`risk-badge ${riskClass(risk)}`}>{risk}</span>
                </div>

                <div className="object-stats">
                  <div className="stat-item">
                    <span className="stat-label">Dist</span>
                    <span className={`stat-value ${obj.estimated_distance_m != null ? 'highlight' : ''}`}>
                      {formatDistance(obj.estimated_distance_m)}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Speed</span>
                    <span className="stat-value">
                      {formatSpeed(obj.closing_speed_mps)}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">TTC</span>
                    <span
                      className={`stat-value ${
                        obj.ttc_seconds != null && obj.ttc_seconds < 2
                          ? 'highlight'
                          : ''
                      }`}
                    >
                      {formatTTC(obj.ttc_seconds)}
                    </span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-label">Conf</span>
                    <span className="stat-value">
                      {(obj.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>

                {obj.in_ego_lane && (
                  <div className="ego-lane-pill" aria-label="In ego lane">
                    ↕ ego lane
                  </div>
                )}
              </article>
            )
          })
        )}
      </div>
    </div>
  )
}
