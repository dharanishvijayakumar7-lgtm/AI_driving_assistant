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

    Args:
        lines:     Output of cv2.HoughLinesP — shape (N, 1, 4) or None.
        min_slope: Minimum |slope| to keep a segment.
        max_slope: Maximum |slope| to keep a segment.

    Returns:
        (left_segments, right_segments) — each is a list of (x1,y1,x2,y2).
    """
    left_segs: list[tuple] = []
    right_segs: list[tuple] = []

    if lines is None:
        return left_segs, right_segs

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue  # perfectly vertical — skip to avoid div-by-zero
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < min_slope or abs(slope) > max_slope:
            continue
        if slope < 0:
            left_segs.append((x1, y1, x2, y2))
        else:
            right_segs.append((x1, y1, x2, y2))

    return left_segs, right_segs


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

    Args:
        segments: List of (x1,y1,x2,y2) tuples for one lane side.
        y_bottom: y coordinate for the lower endpoint (typically frame_height-1).
        y_top:    y coordinate for the upper endpoint (top of ROI).

    Returns:
        (x_bottom, y_bottom, x_top, y_top) in pixel coordinates, or None if
        fitting fails (fewer than 2 unique points).
    """
    if not segments:
        return None

    # Unpack all endpoints into flat coordinate lists
    xs, ys = [], []
    for x1, y1, x2, y2 in segments:
        xs += [x1, x2]
        ys += [y1, y2]

    if len(xs) < 2:
        return None

    try:
        coeffs = np.polyfit(ys, xs, deg=1)   # fit x = m*y + b
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
