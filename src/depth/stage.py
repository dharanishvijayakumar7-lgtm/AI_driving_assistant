"""
stage.py — DepthEstimationStage: FrameProcessor-compatible adapter for
           monocular depth estimation and per-object distance extraction.

This stage MUST run AFTER DetectionStage in the pipeline because it reads
meta["tracked_objects"] to compute per-object distance estimates. It also
reads the tracked objects' bounding boxes to enrich them with an
``estimated_distance_m`` field for downstream stages (Day 6 collision logic).

Pipeline position:
    1. DetectionStage       → populates meta["tracked_objects"]
    2. LaneDetectionStage   → populates meta["lane_lines"], meta["lane_offset"]
    3. DepthEstimationStage → populates meta["depth_map"],
                              enriches each TrackedObject with estimated_distance_m

After this stage runs, the metadata dict carries:
    meta["depth_map"]   = np.ndarray (H, W, 3) — colorized BGR depth heatmap
                          ready for visualization (picture-in-picture overlay).
    Each object in meta["tracked_objects"] gains:
        obj.estimated_distance_m = float  — pseudo-metric distance in meters.
"""

from typing import Any

import numpy as np

from src.depth.depth_estimator import DepthEstimator
from src.depth.depth_utils import (
    colorize_depth_map,
    estimate_object_distance,
    relative_to_pseudo_meters,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DepthEstimationStage:
    """
    FrameProcessor stage that runs monocular depth estimation on every frame,
    then enriches each tracked object with an estimated distance.

    This is the only class that main.py needs to import from the depth package.
    It implements the stage callable interface:

        (frame: np.ndarray, meta: dict) -> (frame: np.ndarray, meta: dict)

    How it hooks into FrameProcessor (one line in main.py):
        processor.add_stage("depth", DepthEstimationStage(config["depth"]))
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize the depth estimator and read visualization config.

        Args:
            config: The ``depth`` sub-dict from config.yaml.
        """
        self._estimator = DepthEstimator(config)
        self._calibration_scale = config.get("calibration_scale", 30.0)
        self._show_heatmap = config.get("show_heatmap_overlay", True)

        # ── Frame-skip: run MiDaS every N frames, reuse cached result ────
        # MiDaS on CPU takes ~1-2 s per frame. Skipping N-1 frames between
        # inference runs is the single biggest FPS lever available without
        # switching to a faster model. depth=3 means ~3x FPS improvement
        # with only slightly stale distances (imperceptible at 1-3 fps).
        self._skip_frames: int = max(1, config.get("skip_frames", 3))
        self._frame_count: int = 0
        self._cached_depth_map: np.ndarray | None = None    # last inference result
        self._cached_depth_colored: np.ndarray | None = None

        logger.info(
            "DepthEstimationStage ready (calibration_scale=%.1f, "
            "show_heatmap=%s, skip_frames=%d).",
            self._calibration_scale,
            self._show_heatmap,
            self._skip_frames,
        )

    def __call__(
        self, frame: np.ndarray, meta: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Run depth estimation and enrich tracked objects with distance.

        Pipeline:
            1. DepthEstimator.estimate()         → raw relative depth map
               (skipped on non-keyframes; cached map reused instead)
            2. For each tracked object:
               a. estimate_object_distance()     → median relative depth in bbox
               b. relative_to_pseudo_meters()    → heuristic metric distance
               c. Attach as obj.estimated_distance_m
            3. Colorize depth map for visualization
            4. Optionally draw picture-in-picture depth heatmap overlay

        Args:
            frame: BGR frame from previous stages (may have detection/lane overlays).
            meta:  Shared metadata dict. MUST contain "tracked_objects" from
                   DetectionStage (list of TrackedObject dataclass instances).

        Returns:
            (frame, meta) — meta now includes "depth_map" and each tracked
            object has an ``estimated_distance_m`` attribute.
        """
        self._frame_count += 1

        # ── 1. Run depth inference (or reuse cache) ───────────────────────
        # Only run the expensive MiDaS model on keyframes; reuse the cached
        # depth map on in-between frames. This is safe because:
        #   - Objects rarely move > 1-2 m between consecutive frames at 1-3 fps.
        #   - The distance is used for TTC estimation which already smooths over
        #     several frames, so a slightly stale depth doesn't matter.
        is_keyframe = (self._frame_count % self._skip_frames == 1)

        if is_keyframe or self._cached_depth_map is None:
            depth_map = self._estimator.estimate(frame)
            depth_colored = colorize_depth_map(depth_map)
            self._cached_depth_map = depth_map
            self._cached_depth_colored = depth_colored
        else:
            # Reuse cached result — no neural network call this frame
            depth_map = self._cached_depth_map
            depth_colored = self._cached_depth_colored

        # 2. Enrich each tracked object with estimated distance
        tracked_objects = meta.get("tracked_objects", [])
        for obj in tracked_objects:
            bbox = (obj.x1, obj.y1, obj.x2, obj.y2)
            rel_depth = estimate_object_distance(depth_map, bbox)
            distance_m = relative_to_pseudo_meters(
                rel_depth, self._calibration_scale
            )
            # Dynamically add the distance field to the TrackedObject dataclass.
            # This is intentional: we extend the Day 2 dataclass without modifying
            # its source file, as required by the Day 4 spec.
            obj.estimated_distance_m = distance_m

        # 3. Store colorized map in meta
        meta["depth_map"] = depth_colored

        # 4. Draw picture-in-picture depth heatmap overlay
        if self._show_heatmap:
            frame = self._draw_pip_overlay(frame, depth_colored)

        # 5. Update bounding box labels to include distance
        frame = self._draw_distance_labels(frame, tracked_objects)

        logger.debug(
            "DepthEstimationStage: %d objects enriched (keyframe=%s).",
            len(tracked_objects), is_keyframe,
        )
        return frame, meta

    @staticmethod
    def _draw_pip_overlay(
        frame: np.ndarray,
        depth_colored: np.ndarray,
        scale: float = 0.25,
        margin: int = 10,
    ) -> np.ndarray:
        """
        Draw a picture-in-picture depth heatmap in the bottom-right corner.

        Args:
            frame:         Main display frame (BGR).
            depth_colored: Colorized depth map (BGR, same size as frame).
            scale:         Size of the PIP relative to the frame (0.25 = 25%).
            margin:        Pixel margin from the frame edge.

        Returns:
            Frame with the PIP overlay drawn.
        """
        fh, fw = frame.shape[:2]
        pip_w = int(fw * scale)
        pip_h = int(fh * scale)

        # Resize depth heatmap to PIP size
        pip = cv2.resize(depth_colored, (pip_w, pip_h), interpolation=cv2.INTER_LINEAR)

        # Position: bottom-right corner
        x1 = fw - pip_w - margin
        y1 = fh - pip_h - margin
        x2 = x1 + pip_w
        y2 = y1 + pip_h

        # Draw a border around the PIP for visual separation
        import cv2 as _cv2
        frame[y1:y2, x1:x2] = pip
        _cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (200, 200, 200), 1)

        # Label
        _cv2.putText(
            frame, "DEPTH",
            (x1 + 4, y1 + 16),
            _cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 1, _cv2.LINE_AA,
        )

        return frame

    @staticmethod
    def _draw_distance_labels(
        frame: np.ndarray,
        tracked_objects: list,
    ) -> np.ndarray:
        """
        Update each tracked object's bounding box label to include distance.

        Draws a small distance tag below the existing label:
            "~18m"

        This is drawn as a separate label to avoid modifying Day 2's
        draw_detections function.

        Args:
            frame:           Display frame with existing detection overlays.
            tracked_objects: List of TrackedObject instances (with estimated_distance_m).

        Returns:
            Frame with distance labels added.
        """
        import cv2 as _cv2

        _FONT = _cv2.FONT_HERSHEY_SIMPLEX
        _SCALE = 0.45
        _PAD = 3

        for obj in tracked_objects:
            distance = getattr(obj, "estimated_distance_m", None)
            if distance is None:
                continue

            # Format distance label
            if distance < 10:
                dist_text = f"~{distance:.1f}m"
            else:
                dist_text = f"~{distance:.0f}m"

            (tw, th), baseline = _cv2.getTextSize(dist_text, _FONT, _SCALE, 1)

            # Position: below the existing label at top of bounding box
            # The Day 2 label sits above y1, so we place the distance tag
            # at the bottom-right of the bounding box for clarity.
            tx = obj.x2 - tw - _PAD * 2
            ty = obj.y2 - _PAD

            # Background pill
            _cv2.rectangle(
                frame,
                (tx - _PAD, ty - th - _PAD),
                (tx + tw + _PAD, ty + baseline + _PAD),
                (0, 0, 0), _cv2.FILLED,
            )
            # Semi-transparent effect
            _cv2.rectangle(
                frame,
                (tx - _PAD, ty - th - _PAD),
                (tx + tw + _PAD, ty + baseline + _PAD),
                (40, 40, 40), 1,
            )

            # Color-code by distance: green = far/safe, yellow = medium, red = close/danger
            if distance > 30:
                color = (0, 210, 70)    # green — safe
            elif distance > 10:
                color = (0, 200, 255)   # yellow — caution
            else:
                color = (0, 60, 255)    # red — danger

            _cv2.putText(
                frame, dist_text,
                (tx, ty),
                _FONT, _SCALE,
                color, 1, _cv2.LINE_AA,
            )

        return frame
