"""
lane_detector.py — Stateful lane detector: Canny → ROI → Hough → smooth.

Classical CV pipeline; no ML models. This is a "v1" approach — accurate on
straight roads with clear lane markings, but limited on curves and at night.
See the LaneDetector docstring for a full list of known limitations.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.lanes.lane_utils import (
    apply_roi_mask,
    separate_lines_by_slope,
    fit_lane_line,
    compute_lane_offset,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default ROI: proportional [x, y] points [bottom-left, bottom-right, top-right, top-left]
_DEFAULT_ROI = [[0.05, 1.0], [0.95, 1.0], [0.60, 0.60], [0.40, 0.60]]


@dataclass
class LaneResult:
    """
    Output of LaneDetector.detect() for one frame.

    All downstream stages (fusion, alerts) should read from this object
    via meta["lane_lines"] / meta["lane_offset"] — never from raw Hough output.

    Attributes:
        left_line:             (x_bottom, y_bottom, x_top, y_top) or None.
        right_line:            (x_bottom, y_bottom, x_top, y_top) or None.
        lane_offset_pixels:    Signed pixel offset of car from lane center.
                               Positive = car is right of center.
        lane_offset_normalized: [-1.0, 1.0] — ±1 means car is at a lane edge.
        left_detected:         True if this frame produced a raw left line fit.
        right_detected:        True if this frame produced a raw right line fit.
    """
    left_line: Optional[tuple[int, int, int, int]] = None
    right_line: Optional[tuple[int, int, int, int]] = None
    lane_offset_pixels: float = 0.0
    lane_offset_normalized: float = 0.0
    left_detected: bool = False
    right_detected: bool = False


class LaneDetector:
    """
    Detects left and right lane lines using classical computer vision.

    Pipeline per frame:
      1. Grayscale + Gaussian blur   (denoise before edge detection)
      2. Canny edge detection        (find intensity gradients)
      3. ROI trapezoid mask          (discard sky, buildings, hood)
      4. Probabilistic Hough lines   (find line segments in edge image)
      5. Slope-based classification  (negative slope → left, positive → right)
      6. Polynomial fit + extrapolation (one line per side, full frame height)
      7. Temporal smoothing          (average last N frames to reduce jitter)

    Known limitations (v1 — classical CV):
      - CURVES: HoughLinesP finds straight segments. On curves, the averaged
        line is a chord through the arc — it drifts off the actual lane edge.
        Fix: replace with polynomial lane fitting (e.g., UFLD model, Day 9).
      - NIGHT / LOW CONTRAST: Canny relies on intensity gradients. Faint or
        worn lane markings produce weak edges that get filtered out.
      - SHADOWS: Strong shadow edges cross the ROI and look like lane lines to
        Hough, causing ghost detections. Adaptive thresholding helps but isn't
        implemented here.
      - STEEP HILLS / TIGHT CURVES: The fixed ROI trapezoid assumes a flat,
        straight road. Uphill slopes move the horizon up; the ROI clips lanes.
      - WET ROADS: Specular reflections create high-contrast edges everywhere,
        flooding the Hough detector with false positives.
      - LANE CHANGES / MULTIPLE LANES: Only the innermost visible lane per
        side is selected (closest to center slope). Adjacent lanes bleed in.
    """

    def __init__(self, config: dict) -> None:
        """
        Read lane config and initialize smoothing buffers.

        Args:
            config: The 'lanes' sub-dict from config.yaml.
        """
        self._canny_low:  int   = config.get("canny_low_threshold",  50)
        self._canny_high: int   = config.get("canny_high_threshold", 150)
        self._blur_k:     int   = config.get("blur_kernel_size", 5)
        self._roi_pts:    list  = config.get("roi_trapezoid", _DEFAULT_ROI)
        self._hough_thr:  int   = config.get("hough_threshold",        30)
        self._hough_min:  int   = config.get("hough_min_line_length",  50)
        self._hough_gap:  int   = config.get("hough_max_line_gap",    100)
        n_smooth:         int   = config.get("smoothing_frames",        5)

        # Each buffer stores (x_bottom, x_top) pairs so we can average
        # endpoints separately without re-extrapolating each time.
        self._left_buf:  deque = deque(maxlen=n_smooth)
        self._right_buf: deque = deque(maxlen=n_smooth)

        logger.info(
            "LaneDetector ready (canny=%d/%d, hough_thr=%d, smooth=%d frames).",
            self._canny_low, self._canny_high, self._hough_thr, n_smooth,
        )

    def detect(self, frame: np.ndarray) -> LaneResult:
        """
        Run the full classical CV pipeline on one frame.

        When no lines are detected on a side, the smoothing buffer provides
        the last-known-good estimate — so brief occlusions don't blank the
        overlay. If the buffer is also empty (first frames), that side
        returns None and the overlay simply omits it.

        Args:
            frame: BGR frame from VideoSource.

        Returns:
            LaneResult with smoothed left/right lines and lane offset.
        """
        h, w = frame.shape[:2]
        y_bottom = h - 1
        y_top    = int(min(p[1] for p in self._roi_pts) * h)

        # ── 1. Preprocess ────────────────────────────────────────────────
        gray     = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred  = cv2.GaussianBlur(gray, (self._blur_k, self._blur_k), 0)

        # ── 2. Canny edges ───────────────────────────────────────────────
        edges = cv2.Canny(blurred, self._canny_low, self._canny_high)

        # ── 3. ROI mask ──────────────────────────────────────────────────
        roi_px = [(int(p[0] * w), int(p[1] * h)) for p in self._roi_pts]
        masked = apply_roi_mask(edges, roi_px)

        # ── 4. Hough transform ───────────────────────────────────────────
        lines = cv2.HoughLinesP(
            masked,
            rho=1,
            theta=np.pi / 180,
            threshold=self._hough_thr,
            minLineLength=self._hough_min,
            maxLineGap=self._hough_gap,
        )

        # ── 5. Classify + fit ────────────────────────────────────────────
        left_segs, right_segs = separate_lines_by_slope(lines)
        left_raw  = fit_lane_line(left_segs,  y_bottom, y_top)
        right_raw = fit_lane_line(right_segs, y_bottom, y_top)

        # ── 6. Update smoothing buffers ──────────────────────────────────
        # Only add to buffer when this frame actually detected the line;
        # stale frames coast on whatever is already in the buffer.
        if left_raw is not None:
            self._left_buf.append((left_raw[0], left_raw[2]))  # (x_bot, x_top)
        if right_raw is not None:
            self._right_buf.append((right_raw[0], right_raw[2]))

        # ── 7. Compute smoothed endpoints ────────────────────────────────
        left_line  = self._smooth_line(self._left_buf,  y_bottom, y_top)
        right_line = self._smooth_line(self._right_buf, y_bottom, y_top)

        left_x  = left_line[0]  if left_line  else None
        right_x = right_line[0] if right_line else None
        offset_px, offset_norm = compute_lane_offset(left_x, right_x, w)

        logger.debug(
            "Lane detect: left=%s right=%s offset=%.2f",
            "OK" if left_line else "—",
            "OK" if right_line else "—",
            offset_norm,
        )
        return LaneResult(
            left_line=left_line,
            right_line=right_line,
            lane_offset_pixels=offset_px,
            lane_offset_normalized=offset_norm,
            left_detected=left_raw is not None,
            right_detected=right_raw is not None,
        )

    # ------------------------------------------------------------------
    def _smooth_line(
        self,
        buf: deque,
        y_bottom: int,
        y_top: int,
    ) -> Optional[tuple[int, int, int, int]]:
        """
        Average buffered (x_bottom, x_top) pairs into one extrapolated line.

        WHY THIS IS NEEDED:
          Hough lines are sensitive to noise — a single pixel of edge can
          shift a detected segment by several pixels. Without smoothing, the
          lane overlay shakes visibly on every frame even when the car isn't
          moving, which is distracting and makes the offset reading noisy.
          Averaging the last N frames is the simplest effective fix; a Kalman
          filter would give smoother results but adds complexity (Day 8).

        Args:
            buf:      Deque of (x_bottom, x_top) from recent frames.
            y_bottom: Fixed y for the lower endpoint.
            y_top:    Fixed y for the upper endpoint.

        Returns:
            (x_bottom, y_bottom, x_top, y_top) or None if buffer is empty.
        """
        if not buf:
            return None
        avg = np.mean(buf, axis=0)
        return (int(avg[0]), y_bottom, int(avg[1]), y_top)

    def reset(self) -> None:
        """Clear smoothing buffers — call when switching video sources."""
        self._left_buf.clear()
        self._right_buf.clear()
        logger.info("LaneDetector smoothing buffers reset.")
