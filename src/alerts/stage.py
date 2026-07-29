"""
stage.py — AlertStage: FrameProcessor-compatible adapter for the alert system
           and final visualization polish.

⚠️  DEPENDENCY CHAIN (this stage MUST run LAST):
    1. DetectionStage       → meta["tracked_objects"]
    2. LaneDetectionStage   → meta["lane_lines"], meta["lane_offset"]
    3. DepthEstimationStage → obj.estimated_distance_m
    4. CollisionFusionStage → obj.risk_level, obj.ttc_seconds, obj.in_ego_lane
    5. AlertStage           → (THIS STAGE) evaluates risk → active_alert banner
                              + final visualization polish

Pipeline position:
    processor.add_stage("detection", ...)  # 1st
    processor.add_stage("lanes",     ...)  # 2nd
    processor.add_stage("depth",     ...)  # 3rd
    processor.add_stage("fusion",    ...)  # 4th
    processor.add_stage("alerts",    ...)  # 5th — LAST

What this stage writes to meta:
    meta["active_alert"]  — ActiveAlert dataclass or None
    meta["viz_config"]    — visualization toggle flags (read by draw_hud)
"""

from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np

from src.alerts.alert_manager import AlertManager, ActiveAlert
from src.fusion.collision_estimator import RISK_DANGER, RISK_CAUTION, RISK_SAFE
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Warning banner constants ──────────────────────────────────────────────────
_BANNER_H       = 56                    # height of the alert banner in pixels
_BANNER_BG      = (10, 10, 10)         # near-black banner background
_BANNER_BG_A    = 0.88                  # opacity
_BANNER_RED     = (0, 40, 220)          # BGR red for DANGER text
_BANNER_YELLOW  = (0, 200, 255)         # BGR yellow for CAUTION text
_FONT_BANNER    = cv2.FONT_HERSHEY_DUPLEX
_FONT_SCALE_BIG = 0.78

# ── Label detail levels ───────────────────────────────────────────────────────
_FONT           = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE_FULL  = 0.42   # full detail: class, dist, speed, TTC, risk
_LABEL_SCALE_DIM   = 0.36   # dimmed: class + ID only for non-critical objects
_LABEL_PAD      = 3

# ── Risk box colors ───────────────────────────────────────────────────────────
_RISK_COLORS: dict[str, tuple[int, int, int]] = {
    RISK_SAFE:    (0, 210, 70),
    RISK_CAUTION: (0, 200, 255),
    RISK_DANGER:  (0, 60, 255),
}
_DIM_COLOR      = (90, 90, 90)          # muted gray for non-critical objects
_DANGER_BOX_T   = 4                     # extra-thick border for in-lane DANGER
_NORMAL_BOX_T   = 2


class AlertStage:
    """
    FrameProcessor stage that:
      1. Runs AlertManager to determine if a debounced DANGER alert is active.
      2. Draws the alert warning banner (with pulse effect when active).
      3. Redraws all bounding-box labels with context-sensitive detail:
           - In-lane CAUTION/DANGER → full detail label (dist, speed, TTC, risk).
           - SAFE or off-lane → minimal, dimmed label (class + ID only).
      4. Writes meta["active_alert"] and meta["viz_config"] for draw_hud().

    Config keys (under ``alerts:`` in config.yaml):
      - danger_persist_frames  (int, default 5)
      - clear_persist_frames   (int, default 10)
      - sound_enabled          (bool, default False)

    Config keys (under ``visualization:`` in config.yaml):
      - show_depth_panel       (bool, default True)
      - show_lane_overlay      (bool, default True)
      - show_fps               (bool, default True)
    """

    def __init__(self, config: dict, viz_config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            The ``alerts`` sub-dict from config.yaml.
        viz_config : dict
            The ``visualization`` sub-dict from config.yaml.
        """
        self._manager = AlertManager(
            danger_persist_frames=config.get("danger_persist_frames", 5),
            clear_persist_frames=config.get("clear_persist_frames", 10),
            sound_enabled=config.get("sound_enabled", False),
        )
        self._viz = {
            "show_depth_panel":   viz_config.get("show_depth_panel",   True),
            "show_lane_overlay":  viz_config.get("show_lane_overlay",  True),
            "show_fps":           viz_config.get("show_fps",           True),
        }

        # Frame counter for pulsing the banner
        self._frame_count: int = 0

        logger.info(
            "AlertStage ready. viz_config=%s", self._viz
        )

    def __call__(
        self, frame: np.ndarray, meta: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Run alert evaluation and draw the final polished visualization layer.

        Pipeline within this stage:
          1. Evaluate AlertManager → active_alert or None
          2. Redraw all object labels with context-sensitive detail/dimming
          3. Draw the warning banner if active_alert is set
          4. Write meta["active_alert"] and meta["viz_config"]
        """
        self._frame_count += 1

        tracked_objects = meta.get("tracked_objects", [])
        logger.debug(
            "[AlertStage] START — frame=%d  objects=%d",
            self._frame_count, len(tracked_objects),
        )

        # ── 1. Evaluate alert state ──────────────────────────────────────
        active_alert = self._manager.evaluate(tracked_objects)

        # ── 2. Redraw bounding boxes with context-sensitive labels ────────
        #    (This pass overdraws the Day 5 labels with the polished version)
        frame = self._draw_labels(frame, tracked_objects, active_alert)

        # ── 3. Draw warning banner if alert is active ─────────────────────
        if active_alert is not None:
            frame = self._draw_banner(frame, active_alert)

        # ── 4. Optionally blank the depth PIP region (demo mode) ─────────
        if not self._viz["show_depth_panel"]:
            frame = self._erase_depth_pip(frame, meta)

        # ── 5. Write metadata ─────────────────────────────────────────────
        meta["active_alert"] = active_alert
        meta["viz_config"] = self._viz

        logger.debug(
            "[AlertStage] END — alert=%s  banner_drawn=%s  labels_redrawn=%d",
            active_alert.severity if active_alert else "None",
            active_alert is not None,
            len(tracked_objects),
        )
        return frame, meta

    # ── Visualization helpers ─────────────────────────────────────────────────

    def _draw_labels(
        self,
        frame: np.ndarray,
        tracked_objects: list,
        active_alert: Optional[ActiveAlert],
    ) -> np.ndarray:
        """
        Draw context-sensitive labels on top of Day 5's boxes.

        Rules:
        - In-lane CAUTION or DANGER: full detail label (distance, speed, TTC,
          risk badge). This is what the driver needs to assess the threat.
        - Everything else (SAFE, or not in ego lane): small, dimmed label
          with just class name + track ID. These are background context —
          they shouldn't compete for attention with the real threats.
        - The alerting object (matching active_alert.track_id) gets an
          extra pulsing inner border to make it unmistakable.
        """
        # Alternate border opacity every 8 frames for a cheap pulse effect
        pulse_bright = (self._frame_count // 8) % 2 == 0

        for obj in tracked_objects:
            risk   = getattr(obj, "risk_level",          RISK_SAFE)
            in_lane = getattr(obj, "in_ego_lane",         False)
            dist   = getattr(obj, "estimated_distance_m", None)
            speed  = getattr(obj, "closing_speed_mps",   None)
            ttc    = getattr(obj, "ttc_seconds",          None)

            # ── Decide label detail level ────────────────────────────────
            is_critical = in_lane and risk in (RISK_CAUTION, RISK_DANGER)
            is_alerting = (
                active_alert is not None
                and obj.track_id == active_alert.track_id
            )

            color = _RISK_COLORS.get(risk, _RISK_COLORS[RISK_SAFE])
            if not is_critical:
                color = _DIM_COLOR

            # ── Box border ───────────────────────────────────────────────
            thickness = _DANGER_BOX_T if (risk == RISK_DANGER and in_lane) else _NORMAL_BOX_T
            cv2.rectangle(frame, (obj.x1, obj.y1), (obj.x2, obj.y2), color, thickness)

            # ── Pulsing inner border for the alerting object ─────────────
            if is_alerting:
                inner_c = (0, 0, 255) if pulse_bright else (60, 60, 200)
                cv2.rectangle(
                    frame,
                    (obj.x1 + 3, obj.y1 + 3),
                    (obj.x2 - 3, obj.y2 - 3),
                    inner_c, 2,
                )

            # ── Build label text ─────────────────────────────────────────
            if is_critical:
                parts = [f"{obj.class_name.capitalize()} #{obj.track_id}"]
                if dist is not None:
                    parts.append(f"~{dist:.0f}m" if dist >= 10 else f"~{dist:.1f}m")
                if speed is not None and speed > 0.1:
                    parts.append(f"{speed:.1f}m/s")
                if ttc is not None:
                    parts.append(f"TTC {ttc:.1f}s")
                parts.append(f"[{risk}]")
                label = "  ".join(parts)
                font_scale = _LABEL_SCALE_FULL
            else:
                # Minimal label for non-critical objects
                label = f"{obj.class_name.capitalize()} #{obj.track_id}"
                font_scale = _LABEL_SCALE_DIM

            # ── Draw label pill ──────────────────────────────────────────
            (tw, th), bl = cv2.getTextSize(label, _FONT, font_scale, 1)
            lx = obj.x1
            ly = max(obj.y1 - _LABEL_PAD, th + _LABEL_PAD * 2)

            # Fill the background only for critical objects; dim gets outline
            if is_critical:
                cv2.rectangle(
                    frame,
                    (lx, ly - th - _LABEL_PAD),
                    (lx + tw + _LABEL_PAD * 2, ly + bl + _LABEL_PAD - 2),
                    color, cv2.FILLED,
                )
                text_color = (255, 255, 255)
            else:
                # Semi-transparent dark pill for dim labels
                pill = frame.copy()
                cv2.rectangle(
                    pill,
                    (lx, ly - th - _LABEL_PAD),
                    (lx + tw + _LABEL_PAD * 2, ly + bl + _LABEL_PAD - 2),
                    (20, 20, 20), cv2.FILLED,
                )
                cv2.addWeighted(pill, 0.55, frame, 0.45, 0, frame)
                text_color = (130, 130, 130)

            cv2.putText(
                frame, label,
                (lx + _LABEL_PAD, ly),
                _FONT, font_scale,
                text_color, 1, cv2.LINE_AA,
            )

        return frame

    def _draw_banner(self, frame: np.ndarray, alert: ActiveAlert) -> np.ndarray:
        """
        Draw a prominent warning banner below the top HUD strip.

        The banner sits just below the HUD (below y=52) so it doesn't collide
        with the FPS/count/lane info row. It pulses by alternating alpha level
        every 6 frames — a low-cost attention signal.

        Layout:
            ┌──────────────────────────────────────────────────┐  ← HUD (y=52)
            │  ⚠  COLLISION RISK — Car #7  |  TTC 1.4s        │  ← Banner
            └──────────────────────────────────────────────────┘
        """
        h, w = frame.shape[:2]

        # HUD height (must match display.py constant)
        HUD_H = 52
        banner_y1 = HUD_H
        banner_y2 = HUD_H + _BANNER_H

        # Pulse: alternate between full and 70% alpha every 6 frames
        pulse = (self._frame_count // 6) % 2 == 0
        alpha = _BANNER_BG_A if pulse else _BANNER_BG_A * 0.72

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, banner_y1), (w, banner_y2), _BANNER_BG, cv2.FILLED)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Bottom border line in red
        cv2.line(frame, (0, banner_y2), (w, banner_y2), _BANNER_RED, 2)

        # ── Banner text ──────────────────────────────────────────────────
        text = alert.message
        (tw, th), _ = cv2.getTextSize(text, _FONT_BANNER, _FONT_SCALE_BIG, 2)

        # Center horizontally, vertically centered in banner
        tx = (w - tw) // 2
        ty = banner_y1 + (_BANNER_H + th) // 2

        # Drop shadow for legibility
        cv2.putText(frame, text, (tx + 2, ty + 2),
                    _FONT_BANNER, _FONT_SCALE_BIG,
                    (0, 0, 0), 3, cv2.LINE_AA)
        # Main text in red
        cv2.putText(frame, text, (tx, ty),
                    _FONT_BANNER, _FONT_SCALE_BIG,
                    _BANNER_RED, 2, cv2.LINE_AA)

        # ── Time-active counter (small, bottom-right of banner) ──────────
        dur_str = f"{alert.seconds_active:.0f}s"
        (dw, dh), _ = cv2.getTextSize(dur_str, _FONT, 0.42, 1)
        cv2.putText(frame, dur_str,
                    (w - dw - 12, banner_y2 - 8),
                    _FONT, 0.42, _BANNER_RED, 1, cv2.LINE_AA)

        return frame

    @staticmethod
    def _erase_depth_pip(frame: np.ndarray, meta: dict) -> np.ndarray:
        """
        In demo mode (show_depth_panel=False), black out the depth PIP region.

        The DepthEstimationStage draws the PIP directly onto the frame before
        this stage runs, so we can't prevent it from drawing — we can only
        cover it. The PIP is always in the bottom-right corner at 25% scale
        with a 10-pixel margin (matching depth/stage.py's _draw_pip_overlay).
        """
        h, w = frame.shape[:2]
        scale  = 0.25
        margin = 10
        pip_w = int(w * scale)
        pip_h = int(h * scale)
        x1 = w - pip_w - margin
        y1 = h - pip_h - margin
        # Fill with a solid dark rectangle to erase the PIP
        cv2.rectangle(frame, (x1 - 2, y1 - 2), (w - margin + 2, h - margin + 2),
                      (18, 18, 18), cv2.FILLED)
        return frame
