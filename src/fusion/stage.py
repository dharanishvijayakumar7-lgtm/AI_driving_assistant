"""
stage.py — CollisionFusionStage: FrameProcessor-compatible adapter for
           temporal fusion, closing speed, TTC, and collision risk.

⚠️  DEPENDENCY CHAIN (this is the first stage that depends on ALL previous):
    1. DetectionStage       → meta["tracked_objects"]  (track IDs, bounding boxes)
    2. LaneDetectionStage   → meta["lane_lines"]       (ego lane boundaries)
    3. DepthEstimationStage → enriches TrackedObject with estimated_distance_m
    4. CollisionFusionStage → (THIS STAGE) reads all of the above, produces
                              closing_speed_mps, ttc_seconds, risk_level,
                              in_ego_lane on each tracked object.

This stage MUST be registered LAST in the pipeline. Moving it before any of
the three upstream stages will produce missing data and incorrect results.

Pipeline position:
    processor.add_stage("detection", ...)  # 1st
    processor.add_stage("lanes",     ...)  # 2nd
    processor.add_stage("depth",     ...)  # 3rd
    processor.add_stage("fusion",    ...)  # 4th — LAST
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from src.fusion.collision_estimator import (
    CollisionEstimator,
    RISK_DANGER,
    RISK_CAUTION,
    RISK_SAFE,
)
from src.fusion.object_history import ObjectHistoryTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Visualization constants ──────────────────────────────────────────────────
_RISK_COLORS: dict[str, tuple[int, int, int]] = {
    RISK_SAFE:    (0, 210, 70),    # green
    RISK_CAUTION: (0, 200, 255),   # yellow
    RISK_DANGER:  (0, 60, 255),    # red
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_SCALE = 0.42
_LABEL_PAD = 3

# Thicker box for in-ego-lane DANGER objects — the single most important signal
_DANGER_EGO_BOX_THICKNESS = 4
_NORMAL_BOX_THICKNESS = 2


class CollisionFusionStage:
    """
    FrameProcessor stage that fuses tracking + depth over time to estimate
    closing speed, TTC, and collision risk for every tracked object.

    After this stage runs, each object in meta["tracked_objects"] gains:
      - obj.closing_speed_mps  (float | None)
      - obj.ttc_seconds        (float | None)
      - obj.risk_level         ("SAFE" | "CAUTION" | "DANGER")
      - obj.in_ego_lane        (bool)

    This is the only class main.py needs to import from the fusion package:
        processor.add_stage("fusion", CollisionFusionStage(config["fusion"]))
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the history tracker and collision estimator from config.

        Expected config keys (under ``fusion:`` in config.yaml):
          - history_length           (int, default 10)
          - history_timeout_seconds  (float, default 2.0)
          - ttc_danger_threshold     (float, default 2.0)
          - ttc_caution_threshold    (float, default 4.0)
          - min_history_points       (int, default 3)
        """
        self._history_tracker = ObjectHistoryTracker(
            max_length=config.get("history_length", 10),
            timeout_seconds=config.get("history_timeout_seconds", 2.0),
        )
        self._estimator = CollisionEstimator(
            ttc_danger_threshold=config.get("ttc_danger_threshold", 2.0),
            ttc_caution_threshold=config.get("ttc_caution_threshold", 4.0),
            min_history_points=config.get("min_history_points", 3),
        )

        # Frame counter for periodic stale-history cleanup
        self._frame_count = 0
        self._cleanup_interval = 30  # every 30 frames

        logger.info("CollisionFusionStage ready.")

    def __call__(
        self, frame: np.ndarray, meta: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Fuse tracking + depth history → closing speed → TTC → risk level.

        Pipeline within this stage:
          1. Read tracked_objects (from DetectionStage + DepthEstimationStage)
          2. Update distance histories with new observations
          3. Expire stale tracks periodically
          4. Run CollisionEstimator to compute risk for each object
          5. Draw risk-coded visualization overlays

        Args:
            frame: BGR frame with existing overlays from prior stages.
            meta:  Shared metadata dict. Must contain "tracked_objects" with
                   estimated_distance_m, and optionally "lane_lines".

        Returns:
            (frame, meta) — meta enriched with risk annotations on each
            tracked object.
        """
        current_time = time.perf_counter()
        self._frame_count += 1

        tracked_objects = meta.get("tracked_objects", [])
        lane_lines = meta.get("lane_lines")

        # ── 1. Update history for each tracked object ────────────────────
        for obj in tracked_objects:
            distance = getattr(obj, "estimated_distance_m", None)
            if distance is not None and obj.track_id >= 0:
                self._history_tracker.update(
                    track_id=obj.track_id,
                    distance_m=distance,
                    timestamp=current_time,
                )

        # ── 2. Expire stale histories periodically ───────────────────────
        if self._frame_count % self._cleanup_interval == 0:
            self._history_tracker.expire_stale(current_time)

        # ── 3. Compute collision risk for all objects ────────────────────
        self._estimator.estimate(
            tracked_objects=tracked_objects,
            object_history=self._history_tracker,
            lane_lines=lane_lines,
        )

        # ── 4. Draw risk visualization ──────────────────────────────────
        frame = self._draw_risk_overlays(frame, tracked_objects)

        logger.debug(
            "CollisionFusionStage: %d objects analyzed, %d active histories.",
            len(tracked_objects),
            self._history_tracker.active_track_count,
        )

        return frame, meta

    @staticmethod
    def _draw_risk_overlays(
        frame: np.ndarray,
        tracked_objects: list,
    ) -> np.ndarray:
        """
        Draw risk-coded bounding box borders and info labels for each object.

        - SAFE objects get green borders and labels.
        - CAUTION objects get yellow borders with closing speed + TTC.
        - DANGER + in_ego_lane objects get thick red borders (the single most
          important visual signal in the whole system).

        This replaces / overdraws the Day 2 default boxes with risk-colored ones.
        """
        for obj in tracked_objects:
            risk_level = getattr(obj, "risk_level", RISK_SAFE)
            in_ego_lane = getattr(obj, "in_ego_lane", False)
            closing_speed = getattr(obj, "closing_speed_mps", None)
            ttc = getattr(obj, "ttc_seconds", None)
            distance = getattr(obj, "estimated_distance_m", None)

            color = _RISK_COLORS.get(risk_level, _RISK_COLORS[RISK_SAFE])

            # ── Box thickness: extra thick for DANGER + in ego lane ──────
            if risk_level == RISK_DANGER and in_ego_lane:
                thickness = _DANGER_EGO_BOX_THICKNESS
            else:
                thickness = _NORMAL_BOX_THICKNESS

            # Draw the risk-colored bounding box (overdraws the detection box)
            cv2.rectangle(
                frame,
                (obj.x1, obj.y1),
                (obj.x2, obj.y2),
                color,
                thickness,
            )

            # ── Build the info label ─────────────────────────────────────
            # Format: "Car #7 ~18m, closing 3.2 m/s, TTC 5.6s [SAFE]"
            parts = [f"{obj.class_name.capitalize()} #{obj.track_id}"]

            if distance is not None:
                if distance < 10:
                    parts.append(f"~{distance:.1f}m")
                else:
                    parts.append(f"~{distance:.0f}m")

            if closing_speed is not None and closing_speed > 0.1:
                parts.append(f"closing {closing_speed:.1f} m/s")

            if ttc is not None:
                parts.append(f"TTC {ttc:.1f}s")

            parts.append(f"[{risk_level}]")

            label = ", ".join(parts)

            # ── Draw label background pill ───────────────────────────────
            (tw, th), baseline = cv2.getTextSize(label, _FONT, _LABEL_SCALE, 1)

            # Position: above the bounding box
            lx = obj.x1
            ly = max(obj.y1 - _LABEL_PAD, th + _LABEL_PAD * 2)

            # Background
            cv2.rectangle(
                frame,
                (lx, ly - th - _LABEL_PAD),
                (lx + tw + _LABEL_PAD * 2, ly + baseline + _LABEL_PAD - 2),
                color,
                cv2.FILLED,
            )

            # Text (white on colored background for readability)
            cv2.putText(
                frame,
                label,
                (lx + _LABEL_PAD, ly),
                _FONT,
                _LABEL_SCALE,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # ── Extra visual emphasis for DANGER + in_ego_lane ───────────
            # Draw a second, inner outline to make it stand out even more
            if risk_level == RISK_DANGER and in_ego_lane:
                # Pulsing effect: alternate intensity via frame count
                # (a cheaper alternative to alpha-blending every frame)
                inner_color = (0, 0, 255)  # pure red inner line
                cv2.rectangle(
                    frame,
                    (obj.x1 + 3, obj.y1 + 3),
                    (obj.x2 - 3, obj.y2 - 3),
                    inner_color,
                    2,
                )

        return frame
