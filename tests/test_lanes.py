"""Tests for lane geometry helpers and the LaneDetector."""

import cv2
import numpy as np

from src.lanes.lane_detector import LaneDetector
from src.lanes.lane_utils import (
    compute_lane_offset,
    fit_lane_line,
    select_innermost_lane,
    separate_lines_by_slope,
)

W, H = 1280, 720
VP = (640.0, 340.0)


def _hough(*segments):
    """Shape segments like cv2.HoughLinesP output: (N, 1, 4)."""
    return np.array([[s] for s in segments], dtype=np.int32)


# ── separate_lines_by_slope ──────────────────────────────────────────────────

def test_slope_sign_decides_side():
    left_seg = (200, 700, 500, 450)    # rises to the right → negative slope
    right_seg = (1080, 700, 780, 450)  # rises to the left  → positive slope
    left, right = separate_lines_by_slope(_hough(left_seg, right_seg))
    assert left == [left_seg]
    assert right == [right_seg]


def test_near_horizontal_segments_rejected():
    left, right = separate_lines_by_slope(_hough((100, 500, 600, 520)), min_slope=0.3)
    assert left == [] and right == []


def test_half_plane_rejects_left_slope_on_right_side():
    # Negative slope, but entirely right of the split: a crack, not a lane.
    crack = (800, 650, 1000, 550)
    left, right = separate_lines_by_slope(_hough(crack), split_x=640)
    assert left == [] and right == []


def test_vanishing_point_filter():
    on_vp = (240, 700, 520, 448)    # extended line passes ~ (640, 340)
    off_vp = (100, 700, 300, 600)   # slope -0.5, misses the VP by ~560 px
    left, _ = separate_lines_by_slope(
        _hough(on_vp, off_vp), split_x=VP[0],
        vanishing_point=VP, vp_tolerance_px=60,
    )
    assert left == [on_vp]


# ── select_innermost_lane ───────────────────────────────────────────────────

def test_innermost_marking_wins_over_adjacent_lane():
    ego = [(300, 719, 450, 560), (380, 634, 470, 539)]        # x≈300 at bottom
    adjacent = [(-200, 719, 200, 560), (0, 640, 250, 540)]    # further out
    chosen = select_innermost_lane(ego + adjacent, y_ref=719, center_x=640,
                                   cluster_tol_px=80, min_support_px=60)
    assert set(chosen) == set(ego)


def test_short_noise_segment_cannot_hijack_selection():
    noise = [(600, 719, 610, 705)]                 # 17 px, innermost
    lane = [(300, 719, 450, 560)]
    chosen = select_innermost_lane(noise + lane, y_ref=719, center_x=640,
                                   cluster_tol_px=80, min_support_px=60)
    assert chosen == lane


# ── fit_lane_line / compute_lane_offset ─────────────────────────────────────

def test_fit_extrapolates_to_requested_rows():
    # x = 0.5 * y on both segments → exact line
    line = fit_lane_line([(300, 600, 325, 650), (340, 680, 350, 700)], 719, 400)
    assert line == (359, 719, 200, 400)


def test_fit_rejects_degenerate_input():
    assert fit_lane_line([], 719, 400) is None
    assert fit_lane_line([(100, 500, 200, 500)], 719, 400) is None   # single row


def test_lane_offset_sign_and_range():
    # Lane centred left of the frame centre → car is right of lane centre
    px, norm = compute_lane_offset(100, 900, W)
    assert px == 140 and 0 < norm <= 1
    assert compute_lane_offset(None, 900, W) == (0.0, 0.0)


# ── LaneDetector on a synthetic road ────────────────────────────────────────

def _synthetic_road(left_bottom_x=240, right_bottom_x=1040):
    """Grey road with two white lane lines converging on VP."""
    img = np.full((H, W, 3), 70, dtype=np.uint8)
    for xb in (left_bottom_x, right_bottom_x):
        # Draw from the bottom up to 55% of the way to the VP
        t = 0.55
        top = (int(xb + (VP[0] - xb) * t), int(H - 1 + (VP[1] - (H - 1)) * t))
        cv2.line(img, (xb, H - 1), top, (255, 255, 255), 8)
    # Distractor: a strong edge that does not point at the VP
    cv2.line(img, (800, 690), (1200, 560), (255, 255, 255), 6)
    return img


def _detector(**overrides):
    cfg = {
        "canny_low_threshold": 50, "canny_high_threshold": 150,
        "roi_polygon": [[0, 1], [1, 1], [1, 0.66], [0.56, 0.52], [0.44, 0.52], [0, 0.66]],
        "vanishing_point": [VP[0] / W, VP[1] / H], "vp_tolerance": 0.06,
        "hough_threshold": 20, "hough_min_line_length": 40, "hough_max_line_gap": 80,
        "min_abs_slope": 0.2, "smoothing_frames": 3, "max_missed_frames": 2,
    }
    cfg.update(overrides)
    return LaneDetector(cfg)


def test_detector_finds_both_lines_without_crossing():
    det = _detector()
    res = det.detect(_synthetic_road())
    assert res.left_line is not None and res.right_line is not None
    lx_bot, _, lx_top, _ = res.left_line
    rx_bot, _, rx_top, _ = res.right_line
    assert abs(lx_bot - 240) < 25 and abs(rx_bot - 1040) < 25
    assert lx_bot < lx_top < VP[0] < rx_top < rx_bot      # converge, never cross
    assert res.ego_source == "detected"
    assert res.lane_offset_normalized is not None


def test_detector_falls_back_to_default_corridor_and_expires_stale_lines():
    det = _detector()
    det.detect(_synthetic_road())
    blank = np.full((H, W, 3), 70, dtype=np.uint8)
    for _ in range(3):                   # > max_missed_frames
        res = det.detect(blank)
    assert res.left_line is None and res.right_line is None
    assert res.ego_source == "default"
    assert res.ego_left is not None and res.ego_right is not None
    assert res.lane_offset_normalized is None
