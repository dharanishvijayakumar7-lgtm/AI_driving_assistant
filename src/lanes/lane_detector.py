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
        raw_hough_count:       Total segments returned by HoughLinesP (before
                               any slope filtering). 0 → Hough found nothing at
                               all; likely ROI misplaced or Canny thresholds too
                               high.
        left_seg_count:        Segments that survived slope filtering as left-
                               lane candidates.
        right_seg_count:       Segments that survived slope filtering as right-
                               lane candidates.
    """
    left_line:              Optional[tuple[int, int, int, int]] = None
    right_line:             Optional[tuple[int, int, int, int]] = None
    lane_offset_pixels:     float = 0.0
    lane_offset_normalized: float = 0.0
    left_detected:          bool  = False
    right_detected:         bool  = False
    # ── Diagnostic counts (populated by detect(), used by stage debug log) ─
    raw_hough_count:        int   = 0
    left_seg_count:         int   = 0
    right_seg_count:        int   = 0


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

        # Diagnostic counts — stored on the result so stage.py can log them
        raw_hough_count  = len(lines) if lines is not None else 0
        left_seg_count   = len(left_segs)
        right_seg_count  = len(right_segs)

        # ── 6. Geometric validation before buffering ─────────────────────
        # ROOT CAUSE OF THE "X-CROSS" BUG:
        # A left lane line projected from y_bottom up to y_top must converge
        # rightward (toward the vanishing point at frame center). Therefore:
        #   x_top > x_bottom  →  VALID left line  (bottom-left, top-right)
        #   x_top < x_bottom  →  INVALID — this is geometrically a right-side
        #                        line that got misclassified (noise segment with
        #                        a marginally negative slope).
        # Symmetric rule for the right line: x_top must be LEFT of x_bottom.
        # Discarding invalid fits prevents them from contaminating the
        # smoothing buffer across multiple frames.
        left_valid  = (left_raw  is not None) and (left_raw[2]  > left_raw[0])
        right_valid = (right_raw is not None) and (right_raw[2] < right_raw[0])

        # ── 7. Update smoothing buffers ──────────────────────────────────
        # Only add to buffer when this frame detected AND validated the line.
        if left_valid:
            self._left_buf.append((left_raw[0], left_raw[2]))   # (x_bot, x_top)
        if right_valid:
            self._right_buf.append((right_raw[0], right_raw[2]))

        # Coordinate logging for the first 3 frames that detect both sides.
        # Enables empirical verification that lines are no longer crossing.
        # Bug-fix audit: left.x_top should be > left.x_bottom;
        #                right.x_top should be < right.x_bottom.
        if not hasattr(self, '_coord_log_count'):
            self._coord_log_count = 0
        if self._coord_log_count < 3 and left_raw is not None and right_raw is not None:
            self._coord_log_count += 1
            logger.info(
                "[Lane coords #%d] "
                "LEFT  raw=(x_bot=%4d, y_bot=%4d, x_top=%4d, y_top=%4d)  valid=%s | "
                "RIGHT raw=(x_bot=%4d, y_bot=%4d, x_top=%4d, y_top=%4d)  valid=%s",
                self._coord_log_count,
                left_raw[0],  left_raw[1],  left_raw[2],  left_raw[3],  left_valid,
                right_raw[0], right_raw[1], right_raw[2], right_raw[3], right_valid,
            )

        # ── 8. Compute smoothed endpoints ────────────────────────────────
        left_line  = self._smooth_line(self._left_buf,  y_bottom, y_top)
        right_line = self._smooth_line(self._right_buf, y_bottom, y_top)

        # ── 9. Inter-line crossing guard ──────────────────────────────────
        # The per-line validation (step 6) ensures each line converges
        # correctly on its own, but independent smoothing buffers can cause
        # the LEFT x_top to drift past the RIGHT x_top — forming an X.
        # Diagnostic logs confirmed this: every frame showed
        #   left_converges=True  right_converges=True  no_cross_top=False
        # The fix: after smoothing, reject the pair if they cross at EITHER
        # endpoint. Clear both buffers so stale data doesn't perpetuate the
        # cross on subsequent frames.
        if left_line is not None and right_line is not None:
            cross_at_top = left_line[2] >= right_line[2]    # left x_top >= right x_top
            cross_at_bot = left_line[0] >= right_line[0]    # left x_bot >= right x_bot
            if cross_at_top or cross_at_bot:
                logger.warning(
                    "[Lane crossing guard] Smoothed lines cross! "
                    "LEFT=(x_bot=%d, x_top=%d)  RIGHT=(x_bot=%d, x_top=%d)  "
                    "Clearing buffers and suppressing this frame.",
                    left_line[0], left_line[2], right_line[0], right_line[2],
                )
                self._left_buf.clear()
                self._right_buf.clear()
                left_line = None
                right_line = None

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
            raw_hough_count=raw_hough_count,
            left_seg_count=left_seg_count,
            right_seg_count=right_seg_count,
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
