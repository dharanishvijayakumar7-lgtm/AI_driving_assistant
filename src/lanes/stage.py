"""
stage.py — LaneDetectionStage: FrameProcessor-compatible adapter for lane detection.

Follows the same (frame, meta) → (frame, meta) contract as Day 2's DetectionStage,
so it plugs into the pipeline with a single processor.add_stage() call in main.py.

Debug mode (lanes.debug_overlay: true in config.yaml):
  Draws the raw Canny edge map and the ROI trapezoid boundary directly onto
  the frame as a semi-transparent overlay so you can visually verify:
    1. Whether Canny is finding any edges at all in the road region.
    2. Whether the ROI trapezoid actually covers the road (vs. sky/hood).
  Every frame also logs at INFO level how many raw Hough segments were found
  and how many survived slope filtering into left/right buckets.
"""

from typing import Any

import numpy as np

from src.lanes.lane_detector import LaneDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LaneDetectionStage:
    """
    Wraps LaneDetector into the FrameProcessor stage interface.

    After this stage executes, the metadata dict carries:
      meta["lane_lines"]  = {"left": (x1,y1,x2,y2) | None,
                              "right": (x1,y1,x2,y2) | None}
      meta["ego_lane"]    = {"left": (x1,y1,x2,y2), "right": (x1,y1,x2,y2),
                              "source": "detected" | "partial" | "default"}
      meta["vanishing_point"] = (x, y) pixel position of the road vanishing point
      meta["road_bottom_y"]   = lowest image row showing road (ROI bottom)
      meta["lane_offset"] = {"pixels": float | None,
                              "normalized": float | None}   # [-1,1], + = right drift

    The fusion stage reads meta["ego_lane"] to decide which objects are in
    the car's path, without re-running lane detection.
    """

    def __init__(self, config: dict) -> None:
        self._detector = LaneDetector(config)
        # Demo-mode toggle: set lanes.show_lane_overlay: false in config.yaml
        # to hide the lane fill/lines for a cleaner recording view.
        self._show_overlay: bool = config.get("show_lane_overlay", True)
        # Debug overlay: draws raw Canny edges + ROI outline so you can see
        # WHY the lane detector is or isn't finding lines. Toggle via
        # lanes.debug_overlay: true in config.yaml.
        self._debug_overlay: bool = config.get("debug_overlay", False)
        self._frame_count: int = 0
        logger.info(
            "LaneDetectionStage ready (show_overlay=%s, debug_overlay=%s).",
            self._show_overlay, self._debug_overlay,
        )

    def __call__(
        self, frame: np.ndarray, meta: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._frame_count += 1
        logger.debug(
            "[LaneDetectionStage] START — frame=%d  show_overlay=%s",
            self._frame_count, self._show_overlay,
        )

        # ── 1. Run detection ─────────────────────────────────────────────
        # Analyse the un-annotated frame — detection boxes already drawn on
        # `frame` would otherwise show up as Canny edges.
        result = self._detector.detect(meta.get("clean_frame", frame))

        # ── 2. Periodic diagnostics (every 30 frames) ────────────────────
        if self._frame_count % 30 == 1:   # frame 1, 31, 61 …
            logger.debug(
                "[LaneDetectionStage] frame=%d  raw_hough=%d  left_segs=%d  "
                "right_segs=%d  left=%s  right=%s  ego=%s",
                self._frame_count, result.raw_hough_count, result.left_seg_count,
                result.right_seg_count, result.left_line, result.right_line,
                result.ego_source,
            )

        # ── 3. Write metadata ────────────────────────────────────────────
        meta["lane_lines"] = {
            "left":  result.left_line,
            "right": result.right_line,
        }
        # The corridor used for in-ego-lane checks. Falls back to default
        # lines through the vanishing point when a side isn't detected, so
        # collision risk stays lane-aware on roads with poor markings.
        meta["ego_lane"] = {
            "left":   result.ego_left,
            "right":  result.ego_right,
            "source": result.ego_source,
        }
        # Horizon/vanishing point — DepthEstimationStage uses its row for
        # ground-plane distance estimation.
        h, w = frame.shape[:2]
        meta["vanishing_point"] = self._detector.vanishing_point_px(w, h)
        # Lowest image row that shows road (bottom of the ROI) — below it is
        # the car's hood, if visible. Ground contact points are clamped to it.
        meta["road_bottom_y"] = self._detector.road_bottom_px(h)
        meta["lane_offset"] = {
            "pixels":     result.lane_offset_pixels,
            "normalized": result.lane_offset_normalized,
        }

        # ── 4. Debug overlay (Canny edges + ROI outline) ─────────────────
        if self._debug_overlay:
            frame = _draw_debug_overlay(frame, self._detector, meta.get("clean_frame", frame))

        # ── 5. Normal lane overlay ───────────────────────────────────────
        if self._show_overlay:
            from src.visualization.display import draw_lane_overlay
            frame = draw_lane_overlay(frame, result)

        logger.debug(
            "[LaneDetectionStage] END — left=%s  right=%s  offset_norm=%s  overlay_drawn=%s",
            "OK" if result.left_line else "NONE",
            "OK" if result.right_line else "NONE",
            result.lane_offset_normalized,
            self._show_overlay and (result.left_line is not None or result.right_line is not None),
        )
        return frame, meta


def _draw_debug_overlay(
    frame: np.ndarray,
    detector: "LaneDetector",
    clean_frame: np.ndarray,
) -> np.ndarray:
    """
    Draw a diagnostic overlay showing:
      - The raw Canny edge map (white edges on dark background) as a
        semi-transparent layer over the bottom half of the frame.
      - The ROI trapezoid boundary in bright cyan so you can see exactly
        what region the lane detector is searching.

    This helps diagnose two of the most common failure modes:
      1. ROI trapezoid is covering the wrong part of the frame (sky/hood).
      2. Canny thresholds are too high — no edges detected inside the ROI.

    Toggle via ``lanes.debug_overlay: true`` in config.yaml.
    """
    import cv2 as _cv2

    h, w = frame.shape[:2]

    # Recompute Canny on the current frame (cheap — just grayscale + blur)
    gray    = _cv2.cvtColor(clean_frame, _cv2.COLOR_BGR2GRAY)
    blurred = _cv2.GaussianBlur(gray, (detector._blur_k, detector._blur_k), 0)
    edges   = _cv2.Canny(blurred, detector._canny_low, detector._canny_high)

    # Convert single-channel edge map to BGR for blending
    edges_bgr = _cv2.cvtColor(edges, _cv2.COLOR_GRAY2BGR)
    # Tint edges green so they're distinct from the regular frame content
    edges_tinted = np.zeros_like(edges_bgr)
    edges_tinted[:, :, 1] = edges  # green channel only

    # Blend at 60% opacity — visible but not blinding
    _cv2.addWeighted(edges_tinted, 0.6, frame, 1.0, 0, frame)

    # Draw the ROI trapezoid boundary in bright cyan
    roi_px = [
        (int(p[0] * w), int(p[1] * h))
        for p in detector._roi_pts
    ]
    pts = np.array(roi_px, dtype=np.int32).reshape((-1, 1, 2))
    _cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # Vanishing point (magenta cross) — lane markings should meet here
    vp_x, vp_y = (int(v) for v in detector.vanishing_point_px(w, h))
    _cv2.drawMarker(frame, (vp_x, vp_y), (255, 0, 255), _cv2.MARKER_CROSS, 24, 2)

    # Label each ROI corner with its config fraction so you can cross-check
    for i, ((px, py), rp) in enumerate(zip(roi_px, detector._roi_pts)):
        label = f"({rp[0]:.2f},{rp[1]:.2f})"
        _cv2.putText(
            frame, label, (px + 4, py - 4),
            _cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1, _cv2.LINE_AA,
        )

    # Small legend in top-left (below the HUD strip)
    _cv2.putText(
        frame, "[LANE DEBUG] green=Canny  cyan=ROI  magenta=vanishing point",
        (10, 75), _cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 200), 1, _cv2.LINE_AA,
    )

    return frame
