"""
stage.py — CollisionFusionStage: FrameProcessor-compatible adapter for
           temporal fusion, closing speed, TTC, and collision risk.

⚠️  DEPENDENCY CHAIN (this is the first stage that depends on ALL previous):
    1. DetectionStage       → meta["tracked_objects"]  (track IDs, bounding boxes)
    2. LaneDetectionStage   → meta["ego_lane"]         (ego lane corridor)
    3. DepthEstimationStage → enriches TrackedObject with estimated_distance_m
    4. CollisionFusionStage → (THIS STAGE) reads all of the above, produces
                              closing_speed_mps, ttc_seconds, risk_level,
                              in_ego_lane on each tracked object.

This stage MUST run after all three upstream stages (and before AlertStage).
Moving it earlier will produce missing data and incorrect results.

Pipeline position:
    processor.add_stage("detection", ...)  # 1st
    processor.add_stage("lanes",     ...)  # 2nd
    processor.add_stage("depth",     ...)  # 3rd
    processor.add_stage("fusion",    ...)  # 4th
    processor.add_stage("alerts",    ...)  # 5th — draws the risk overlay
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.fusion.collision_estimator import CollisionEstimator
from src.fusion.object_history import ObjectHistoryTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
            max_closing_speed_mps=config.get("max_closing_speed_mps", 40.0),
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

        Args:
            frame: BGR frame with existing overlays from prior stages.
            meta:  Shared metadata dict. Must contain "tracked_objects" with
                   estimated_distance_m, and optionally "lane_lines".

        Returns:
            (frame, meta) — meta enriched with risk annotations on each
            tracked object.
        """
        # Frame capture time (video time for files) — NOT processing time.
        current_time = meta.get("timestamp", time.perf_counter())
        self._frame_count += 1

        tracked_objects = meta.get("tracked_objects", [])
        # The ego corridor always has both sides (detected or default lines);
        # fall back to raw lane lines if the lane stage predates it.
        lane_lines = meta.get("ego_lane") or meta.get("lane_lines")

        logger.debug(
            "[CollisionFusionStage] START — frame=%d  objects=%d  lane_lines=%s",
            self._frame_count,
            len(tracked_objects),
            "present" if lane_lines and (lane_lines.get("left") or lane_lines.get("right")) else "absent",
        )

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

        # Frame is returned untouched: AlertStage draws the risk-coloured
        # boxes and labels, so they are not drawn twice.

        risk_counts = {}
        for obj in tracked_objects:
            r = getattr(obj, "risk_level", "SAFE")
            risk_counts[r] = risk_counts.get(r, 0) + 1
        logger.debug(
            "[CollisionFusionStage] END — risk_summary=%s  histories=%d",
            risk_counts,
            self._history_tracker.active_track_count,
        )

        return frame, meta
