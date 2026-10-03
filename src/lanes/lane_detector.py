"""
lane_detector.py — Stateful lane detector: Canny → ROI → Hough → smooth.

Classical CV pipeline; no ML models. This is a "v1" approach — accurate on
straight roads with clear lane markings, but limited on curves and at night.
See the LaneDetector docstring for a full list of known limitations.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.lanes.lane_utils import (
    apply_roi_mask,
    separate_lines_by_slope,
    select_innermost_lane,
    fit_lane_line,
    compute_lane_offset,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default ROI: proportional [x, y] points [bottom-left, bottom-right, top-right, top-left]
_DEFAULT_ROI = [[0.05, 1.0], [0.95, 1.0], [0.60, 0.60], [0.40, 0.60]]

LaneLine = tuple[int, int, int, int]   # (x_bottom, y_bottom, x_top, y_top)


@dataclass
class LaneResult:
    """
    Output of LaneDetector.detect() for one frame.

    All downstream stages (fusion, alerts) should read from this object
    via meta["lane_lines"] / meta["ego_lane"] — never from raw Hough output.

    Attributes:
        left_line:             (x_bottom, y_bottom, x_top, y_top) or None.
        right_line:            (x_bottom, y_bottom, x_top, y_top) or None.
        ego_left / ego_right:  The ego-lane corridor used for in-lane checks.
                               Equal to the detected line when available,
                               otherwise a default line through the vanishing
                               point (see ``default_lane_bottom_x``).
        ego_source:            "detected" (both sides), "partial" (one side)
                               or "default" (neither side detected).
        lane_offset_pixels:    Signed pixel offset of car from lane center.
                               Positive = car is right of center.
        lane_offset_normalized: [-1.0, 1.0] — ±1 means car is at a lane edge.
                               None when both lines are not available.
        left_detected:         True if this frame produced a raw left line fit.
        right_detected:        True if this frame produced a raw right line fit.
        raw_hough_count:       Total segments returned by HoughLinesP (before
                               any filtering). 0 → Hough found nothing at all;
                               likely ROI misplaced or Canny thresholds too high.
        left_seg_count:        Segments that survived filtering as left-lane
                               candidates.
        right_seg_count:       Segments that survived filtering as right-lane
                               candidates.
    """
    left_line:              Optional[LaneLine] = None
    right_line:             Optional[LaneLine] = None
    ego_left:               Optional[LaneLine] = None
    ego_right:              Optional[LaneLine] = None
    ego_source:             str   = "default"
    lane_offset_pixels:     Optional[float] = None
    lane_offset_normalized: Optional[float] = None
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
      3. ROI polygon mask            (discard sky, buildings, hood)
      4. Probabilistic Hough lines   (find line segments in edge image)
      5. Geometric filtering         (slope sign, half-plane, VP consistency)
      6. Innermost-marking selection (ego-lane boundary, not adjacent lanes)
      7. Weighted line fit + extrapolation to the ROI's top and bottom rows
      8. Temporal smoothing          (average last N frames to reduce jitter)

    The vanishing point (``lanes.vanishing_point``) anchors steps 5-6 and is
    camera-specific: it is where the lane markings meet on the horizon, as a
    fraction of frame width/height. Tune it (and ``roi_polygon``) once per
    camera mounting — see config.yaml for presets.

    Known limitations (v1 — classical CV):
      - CURVES: HoughLinesP finds straight segments. On curves, the averaged
        line is a chord through the arc — it drifts off the actual lane edge.
        Fix: polynomial lane fitting or an ML lane model (e.g. UFLD).
      - NIGHT / LOW CONTRAST: Canny relies on intensity gradients. Faint or
        worn lane markings produce weak edges that get filtered out.
      - SHADOWS: Strong shadow edges that happen to point at the vanishing
        point still look like lane lines.
      - HILLS: The fixed vanishing point assumes a flat road. Up/down slopes
        move the horizon and the VP filter starts rejecting real markings.
      - WET ROADS: Specular reflections create high-contrast edges everywhere,
        flooding the Hough detector with false positives.
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
        # roi_polygon (any number of points) supersedes the older 4-point
        # roi_trapezoid key; both are lists of [x_frac, y_frac].
        self._roi_pts:    list  = config.get(
            "roi_polygon", config.get("roi_trapezoid", _DEFAULT_ROI)
        )
        self._hough_thr:  int   = config.get("hough_threshold",        30)
        self._hough_min:  int   = config.get("hough_min_line_length",  50)
        self._hough_gap:  int   = config.get("hough_max_line_gap",    100)
        self._min_slope:  float = config.get("min_abs_slope", 0.3)
        self._vp_frac:    Optional[list] = config.get("vanishing_point")
        self._vp_tol:     float = config.get("vp_tolerance", 0.08)
        self._cluster_tol: float = config.get("lane_cluster_tolerance", 0.08)
        self._default_bottom_x: list = config.get("default_lane_bottom_x", [0.15, 0.85])
        self._max_missed: int   = config.get("max_missed_frames", 15)
        n_smooth:         int   = config.get("smoothing_frames",        5)

        # Each buffer stores (x_bottom, x_top) pairs so we can average
        # endpoints separately without re-extrapolating each time.
        self._left_buf:  deque = deque(maxlen=n_smooth)
        self._right_buf: deque = deque(maxlen=n_smooth)
        # Consecutive frames without a valid detection, per side. When a side
        # stays undetected for max_missed_frames its buffer is dropped, so a
        # stale line doesn't linger on screen (and in the in-lane check).
        self._left_missed:  int = 0
        self._right_missed: int = 0

        logger.info(
            "LaneDetector ready (canny=%d/%d, hough_thr=%d, smooth=%d frames, vp=%s).",
            self._canny_low, self._canny_high, self._hough_thr, n_smooth, self._vp_frac,
        )

    def detect(self, frame: np.ndarray) -> LaneResult:
        """
        Run the full classical CV pipeline on one frame.

        When no lines are detected on a side, the smoothing buffer provides
        the last-known-good estimate for up to ``max_missed_frames`` frames —
        so brief gaps between dashed markings don't blank the overlay.

        Args:
            frame: BGR frame from VideoSource.

        Returns:
            LaneResult with smoothed left/right lines, ego corridor and offset.
        """
        h, w = frame.shape[:2]
        vp_x, vp_y = self.vanishing_point_px(w, h)
        y_bottom = self.road_bottom_px(h)
        # Lines are extrapolated up to the ROI top, but never to (or past)
        # the vanishing point, where left and right meet and would cross.
        y_top = max(int(min(p[1] for p in self._roi_pts) * h), int(vp_y + 0.03 * h))

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

        # ── 5. Classify (slope sign + half-plane + VP consistency) ──────
        left_segs, right_segs = separate_lines_by_slope(
            lines,
            min_slope=self._min_slope,
            split_x=vp_x,
            vanishing_point=(vp_x, vp_y) if self._vp_frac else None,
            vp_tolerance_px=self._vp_tol * w,
        )

        # ── 6. Keep only the innermost marking per side ─────────────────
        cluster_tol = self._cluster_tol * w
        min_support = 1.5 * self._hough_min
        left_segs  = select_innermost_lane(left_segs,  y_bottom, vp_x, cluster_tol, min_support)
        right_segs = select_innermost_lane(right_segs, y_bottom, vp_x, cluster_tol, min_support)

        # ── 7. Fit ──────────────────────────────────────────────────────
        left_raw  = fit_lane_line(left_segs,  y_bottom, y_top)
        right_raw = fit_lane_line(right_segs, y_bottom, y_top)

        # A left line must converge rightward toward the VP (x_top > x_bottom)
        # and stay left of it; mirror rule on the right. Anything else is a
        # misfit and must not enter the smoothing buffer.
        left_valid  = left_raw  is not None and left_raw[0]  < left_raw[2]  < vp_x
        right_valid = right_raw is not None and right_raw[0] > right_raw[2] > vp_x

        # ── 8. Update smoothing buffers ──────────────────────────────────
        if left_valid:
            self._left_buf.append((left_raw[0], left_raw[2]))   # (x_bot, x_top)
            self._left_missed = 0
        else:
            self._left_missed += 1
            if self._left_missed > self._max_missed:
                self._left_buf.clear()
        if right_valid:
            self._right_buf.append((right_raw[0], right_raw[2]))
            self._right_missed = 0
        else:
            self._right_missed += 1
            if self._right_missed > self._max_missed:
                self._right_buf.clear()

        left_line  = self._smooth_line(self._left_buf,  y_bottom, y_top)
        right_line = self._smooth_line(self._right_buf, y_bottom, y_top)

        # ── 9. Ego corridor + lane offset ────────────────────────────────
        ego_left  = left_line  or self._default_line(self._default_bottom_x[0] * w,
                                                     vp_x, vp_y, y_bottom, y_top)
        ego_right = right_line or self._default_line(self._default_bottom_x[1] * w,
                                                     vp_x, vp_y, y_bottom, y_top)
        n_detected = (left_line is not None) + (right_line is not None)
        ego_source = ("default", "partial", "detected")[n_detected]

        offset_px: Optional[float] = None
        offset_norm: Optional[float] = None
        if left_line and right_line:
            offset_px, offset_norm = compute_lane_offset(left_line[0], right_line[0], w)

        logger.debug(
            "Lane detect: left=%s right=%s offset=%s",
            "OK" if left_line else "—",
            "OK" if right_line else "—",
            offset_norm,
        )
        return LaneResult(
            left_line=left_line,
            right_line=right_line,
            ego_left=ego_left,
            ego_right=ego_right,
            ego_source=ego_source,
            lane_offset_pixels=offset_px,
            lane_offset_normalized=offset_norm,
            left_detected=left_valid,
            right_detected=right_valid,
            raw_hough_count=len(lines) if lines is not None else 0,
            left_seg_count=len(left_segs),
            right_seg_count=len(right_segs),
        )

    # ------------------------------------------------------------------
    def vanishing_point_px(self, w: int, h: int) -> tuple[float, float]:
        """
        Pixel position of the configured vanishing point. Without one, assume
        the frame's horizontal centre, slightly above the ROI's top edge.
        """
        if self._vp_frac:
            return self._vp_frac[0] * w, self._vp_frac[1] * h
        return w / 2.0, (min(p[1] for p in self._roi_pts) - 0.05) * h

    def road_bottom_px(self, h: int) -> int:
        """Lowest image row showing road: the ROI's bottom edge (hood above it)."""
        return min(int(max(p[1] for p in self._roi_pts) * h), h - 1)

    @staticmethod
    def _default_line(
        x_bottom: float,
        vp_x: float,
        vp_y: float,
        y_bottom: int,
        y_top: int,
    ) -> LaneLine:
        """Straight line from (x_bottom, y_bottom) toward the vanishing point."""
        t = (y_top - vp_y) / max(y_bottom - vp_y, 1e-6)
        x_top = vp_x + (x_bottom - vp_x) * t
        return (int(x_bottom), y_bottom, int(x_top), y_top)

    def _smooth_line(
        self,
        buf: deque,
        y_bottom: int,
        y_top: int,
    ) -> Optional[LaneLine]:
        """
        Average buffered (x_bottom, x_top) pairs into one extrapolated line.

        WHY THIS IS NEEDED:
          Hough lines are sensitive to noise — a single pixel of edge can
          shift a detected segment by several pixels. Without smoothing, the
          lane overlay shakes visibly on every frame even when the car isn't
          moving, which is distracting and makes the offset reading noisy.
          Averaging the last N frames is the simplest effective fix; a Kalman
          filter would give smoother results but adds complexity.

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
        self._left_missed = self._right_missed = 0
        logger.info("LaneDetector smoothing buffers reset.")
