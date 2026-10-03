"""Tests for distance estimation helpers."""

import numpy as np
import pytest

from src.depth.depth_utils import (
    estimate_object_distance,
    focal_length_px,
    ground_plane_distance,
    relative_to_pseudo_meters,
)


def test_focal_length_from_fov():
    # 90° HFOV → f = half the width
    assert focal_length_px(1280, 90.0) == pytest.approx(640.0)


def test_ground_plane_distance_matches_pinhole_geometry():
    f, cam_h, horizon = 1000.0, 1.2, 340.0
    # A point 20 m ahead projects to horizon + f*H/Z = 340 + 60 = 400
    assert ground_plane_distance(400, horizon, f, cam_h) == pytest.approx(20.0)
    # Closer objects sit lower in the image
    assert ground_plane_distance(640, horizon, f, cam_h) == pytest.approx(4.0)


def test_ground_plane_distance_rejects_horizon_and_clamps():
    assert ground_plane_distance(341, 340, 1000, 1.2) is None    # < 3 px below
    assert ground_plane_distance(300, 340, 1000, 1.2) is None    # above horizon
    assert ground_plane_distance(344, 340, 1000, 1.2, max_distance_m=200) == 200


def test_midas_object_depth_uses_robust_median():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[20:80, 20:80] = 0.5          # object
    depth[20:30, 20:80] = 1.0          # bright edge band (background leak)
    assert estimate_object_distance(depth, (20, 20, 80, 80)) == pytest.approx(0.5)


def test_pseudo_meters_is_inverse_and_clamped():
    assert relative_to_pseudo_meters(0.5, 8.0) == pytest.approx(16.0, rel=1e-3)
    assert relative_to_pseudo_meters(0.0, 8.0) == 200.0
