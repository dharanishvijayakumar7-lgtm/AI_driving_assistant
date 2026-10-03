"""
lane_utils.py — Pure helper functions for the lane detection pipeline.

Keeping these stateless and importable in isolation makes them easy to
unit-test without spinning up a full LaneDetector.
"""

from typing import Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# ROI masking
# ---------------------------------------------------------------------------

def apply_roi_mask(edges: np.ndarray, roi_pts: list[tuple[int, int]]) -> np.ndarray:
    """
    Zero out everything in `edges` outside the given trapezoid polygon.

    Args:
        edges:   A single-channel (grayscale/binary) image.
        roi_pts: Pixel-coordinate vertices of the ROI polygon, ordered as
                 [bottom-left, bottom-right, top-right, top-left].

    Returns:
        A copy of `edges` with pixels outside the polygon set to 0.
    """
    mask = np.zeros_like(edges)
    pts = np.array(roi_pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return cv2.bitwise_and(edges, mask)


# ---------------------------------------------------------------------------
# Slope-based line classification
# ---------------------------------------------------------------------------

def separate_lines_by_slope(
    lines: Optional[np.ndarray],
    min_slope: float = 0.3,
    max_slope: float = 10.0,
    split_x: Optional[float] = None,
    vanishing_point: Optional[tuple[float, float]] = None,
    vp_tolerance_px: Optional[float] = None,
) -> tuple[list, list]:
    """
    Split raw HoughLinesP segments into left-lane and right-lane candidates.

    WHY SLOPE DETERMINES SIDE — in image coordinates (origin = top-left,
    y increases *downward*, x increases rightward):

      Left lane lines converge from the bottom-left toward the vanishing
      point at the upper-center of the frame.  Moving along such a line
      from bottom to top: x *increases* (rightward) while y *decreases*
      (upward).  slope = Δy / Δx = negative / positive = **NEGATIVE**.

      Right lane lines converge from the bottom-right toward the same
      vanishing point.  Moving along such a line from bottom to top:
      x *decreases* (leftward) while y *decreases* (upward).
      slope = Δy / Δx = negative / negative = **POSITIVE**.

    Lines with |slope| < min_slope are near-horizontal noise (road cracks,
    horizon).  Lines with |slope| > max_slope are near-vertical noise
    (lamp-posts, road edges).  Both are discarded.

    Slope alone is not enough, though: a crack or shadow edge on the RIGHT
    half of the road with a negative slope gets filed as a "left" segment and
    drags the fitted left line across the frame (the X-cross bug). Two
    geometric filters remove almost all of that noise:

      - Half-plane split: a left-lane segment must lie entirely left of
        ``split_x`` (the vanishing point's x), a right-lane segment entirely
        right of it.
      - Vanishing-point consistency: every real lane line passes close to the
        vanishing point. A segment whose extended line misses the VP row by
        more than ``vp_tolerance_px`` horizontally is not a lane marking.

    Args:
        lines:           Output of cv2.HoughLinesP — shape (N, 1, 4) or None.
        min_slope:       Minimum |slope| to keep a segment.
        max_slope:       Maximum |slope| to keep a segment.
        split_x:         x dividing left from right candidates (None = off).
        vanishing_point: (x, y) pixel position of the road vanishing point
                         (None disables the VP-consistency filter).
        vp_tolerance_px: Max horizontal miss distance at the VP row.

    Returns:
        (left_segments, right_segments) — each is a list of (x1,y1,x2,y2).
    """
    left_segs: list[tuple] = []
    right_segs: list[tuple] = []

    if lines is None:
        return left_segs, right_segs

    for line in lines:
        # HoughLinesP output shape is nominally (N, 1, 4) but some OpenCV
        # builds — especially on large frames — return (N, 4) directly.
        # flatten()[:4] handles both shapes without an explicit reshape.
        x1, y1, x2, y2 = (int(v) for v in line.flatten()[:4])
        if x2 == x1:
            continue  # perfectly vertical — skip to avoid div-by-zero
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < min_slope or abs(slope) > max_slope:
            continue

        is_left = slope < 0
        if split_x is not None:
            if is_left and max(x1, x2) > split_x:
                continue
            if not is_left and min(x1, x2) < split_x:
                continue

        if vanishing_point is not None and vp_tolerance_px is not None:
            vp_x, vp_y = vanishing_point
            x_at_vp = x1 + (vp_y - y1) / slope   # extended line at the VP row
            if abs(x_at_vp - vp_x) > vp_tolerance_px:
                continue

        if is_left:
            left_segs.append((x1, y1, x2, y2))
        else:
            right_segs.append((x1, y1, x2, y2))

    return left_segs, right_segs


def select_innermost_lane(
    segments: list[tuple],
    y_ref: float,
    center_x: float,
    cluster_tol_px: float,
    min_support_px: float,
) -> list[tuple]:
    """
    Keep only the segments belonging to the lane marking closest to the car.

    On a multi-lane road one side can contain several markings (the ego-lane
    boundary plus adjacent-lane boundaries). Fitting a single line through all
    of them produces a line that matches none. Each segment is projected to
    row ``y_ref`` (the ROI bottom); segments within ``cluster_tol_px`` of each
    other there belong to the same marking. The innermost cluster whose total
    segment length reaches ``min_support_px`` wins, so one short noise
    fragment near the centre cannot hijack the selection.

    Args:
        segments:       (x1,y1,x2,y2) tuples for one side.
        y_ref:          Row at which segments are compared.
        center_x:       x of the car's heading (vanishing point x).
        cluster_tol_px: Max spread of one marking at y_ref.
        min_support_px: Minimum total segment length for a cluster to count.

    Returns:
        The selected subset of ``segments`` (empty if none qualifies).
    """
    projected = []
    for seg in segments:
        x1, y1, x2, y2 = seg
        if y2 == y1:
            continue
        x_ref = x1 + (y_ref - y1) * (x2 - x1) / (y2 - y1)
        length = float(np.hypot(x2 - x1, y2 - y1))
        projected.append((abs(x_ref - center_x), x_ref, length, seg))

    projected.sort(key=lambda p: p[0])   # innermost first

    for _, anchor_x, _, _ in projected:
        cluster = [p for p in projected if abs(p[1] - anchor_x) <= cluster_tol_px]
        if sum(p[2] for p in cluster) >= min_support_px:
            return [p[3] for p in cluster]

    return []


# ---------------------------------------------------------------------------
# Line fitting and extrapolation
# ---------------------------------------------------------------------------

def fit_lane_line(
    segments: list[tuple],
    y_bottom: int,
    y_top: int,
) -> Optional[tuple[int, int, int, int]]:
    """
    Fit one representative line through a collection of segments and
    extrapolate it to span from y_bottom (bottom of frame) to y_top (horizon).

    WHY WE FIT x = f(y) INSTEAD OF y = f(x):
      We want to answer "where is the lane at row y?" for two fixed y values
      (bottom of frame, top of ROI).  Fitting x as a function of y lets us
      directly query poly(y_bottom) and poly(y_top) for the lane's x position.
      Fitting y = f(x) would require inverting the function, and fails for
      near-vertical lines where the slope is very steep.

    Endpoints are weighted by their segment's length, so one long, clean
    lane-marking segment outweighs several short noise fragments.

    Args:
        segments: List of (x1,y1,x2,y2) tuples for one lane side.
        y_bottom: y coordinate for the lower endpoint (bottom of the ROI).
        y_top:    y coordinate for the upper endpoint (top of ROI).

    Returns:
        (x_bottom, y_bottom, x_top, y_top) in pixel coordinates, or None if
        fitting fails (fewer than 2 unique points).
    """
    if not segments:
        return None

    # Unpack all endpoints into flat coordinate lists
    xs, ys, ws = [], [], []
    for x1, y1, x2, y2 in segments:
        # polyfit squares the weights, so sqrt(length) weights by length
        w = float(np.sqrt(max(np.hypot(x2 - x1, y2 - y1), 1.0)))
        xs += [x1, x2]
        ys += [y1, y2]
        ws += [w, w]

    if len(set(ys)) < 2:
        return None

    try:
        coeffs = np.polyfit(ys, xs, deg=1, w=ws)   # fit x = m*y + b
        poly = np.poly1d(coeffs)
        x_bottom = int(np.clip(poly(y_bottom), -5000, 5000))
        x_top    = int(np.clip(poly(y_top),    -5000, 5000))
        return (x_bottom, y_bottom, x_top, y_top)
    except (np.linalg.LinAlgError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Lane center offset
# ---------------------------------------------------------------------------

def compute_lane_offset(
    left_x_bottom: Optional[int],
    right_x_bottom: Optional[int],
    frame_width: int,
) -> tuple[float, float]:
    """
    Compute how far the camera (car center) is from the lane midpoint.

    Convention:
      - Positive offset → car is to the RIGHT of lane center.
      - Negative offset → car is to the LEFT of lane center.

    The pixel offset is normalized by half the lane width so ±1.0 means
    the car's center is exactly at a lane edge — a sensor-fusion-friendly
    representation that doesn't depend on absolute pixel counts.

    Args:
        left_x_bottom:  x of the left lane at the bottom of frame, or None.
        right_x_bottom: x of the right lane at the bottom of frame, or None.
        frame_width:    Frame width in pixels.

    Returns:
        (offset_pixels, offset_normalized) — both 0.0 if lanes unavailable.
    """
    if left_x_bottom is None or right_x_bottom is None:
        return 0.0, 0.0

    lane_center = (left_x_bottom + right_x_bottom) / 2.0
    frame_center = frame_width / 2.0
    offset_px = frame_center - lane_center   # positive = car right of center

    lane_width = right_x_bottom - left_x_bottom
    if lane_width <= 0:
        return float(offset_px), 0.0

    offset_norm = offset_px / (lane_width / 2.0)
    offset_norm = float(np.clip(offset_norm, -1.0, 1.0))
    return float(offset_px), offset_norm
