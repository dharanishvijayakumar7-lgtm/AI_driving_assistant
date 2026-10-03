"""Tests for TTC / risk estimation, ego-lane membership and alert hysteresis."""

from dataclasses import dataclass
from typing import Optional

import pytest

from src.alerts.alert_manager import AlertManager
from src.fusion.collision_estimator import CollisionEstimator
from src.fusion.object_history import ObjectHistoryTracker


@dataclass
class Obj:
    """Minimal stand-in for detection.stage.TrackedObject."""
    track_id: int
    x1: int = 600
    y1: int = 400
    x2: int = 680
    y2: int = 470
    class_name: str = "car"
    estimated_distance_m: Optional[float] = None


# Ego corridor from (200, 719)/(1080, 719) up to the vanishing point (640, 340)
EGO = {"left": (200, 719, 527, 437), "right": (1080, 719, 753, 437)}


def _history(track_id, distances, fps=24.0):
    hist = ObjectHistoryTracker(max_length=10, timeout_seconds=2.0)
    for i, d in enumerate(distances):
        hist.update(track_id, d, timestamp=i / fps)
    return hist


def test_closing_object_gets_ttc_and_danger():
    # 20 m closing at 12 m/s, sampled at 24 fps
    dists = [20.0 - 12.0 * i / 24 for i in range(10)]
    obj = Obj(track_id=1)
    CollisionEstimator(2.0, 4.0, 3).estimate([obj], _history(1, dists), EGO)
    assert obj.closing_speed_mps == pytest.approx(12.0, abs=0.05)
    assert obj.ttc_seconds == pytest.approx(dists[-1] / 12.0, abs=0.1)
    assert obj.in_ego_lane is True
    assert obj.risk_level == "DANGER"


def test_receding_object_is_safe_with_no_ttc():
    obj = Obj(track_id=2)
    CollisionEstimator().estimate([obj], _history(2, [10, 11, 12, 13, 14]), EGO)
    assert obj.ttc_seconds is None
    assert obj.risk_level == "SAFE"


def test_insufficient_history_is_safe():
    obj = Obj(track_id=3)
    CollisionEstimator(min_history_points=3).estimate([obj], _history(3, [10, 5]), EGO)
    assert obj.ttc_seconds is None and obj.risk_level == "SAFE"


def test_off_lane_risk_is_downgraded():
    dists = [20.0 - 12.0 * i / 24 for i in range(10)]
    obj = Obj(track_id=4, x1=60, x2=160, y2=600)   # bottom-centre left of the lane
    CollisionEstimator(2.0, 4.0, 3).estimate([obj], _history(4, dists), EGO)
    assert obj.in_ego_lane is False
    assert obj.risk_level == "CAUTION"


def test_ego_lane_uses_ground_contact_point():
    # Wide truck box: centre row would be far up the image, bottom is on the road
    truck = Obj(track_id=5, x1=560, y1=200, x2=720, y2=520)
    CollisionEstimator().estimate([truck], ObjectHistoryTracker(), EGO)
    assert truck.in_ego_lane is True


def test_objects_beyond_the_horizon_are_not_in_lane():
    far = Obj(track_id=6, x1=630, y1=320, x2=650, y2=335)   # above the VP row
    CollisionEstimator().estimate([far], ObjectHistoryTracker(), EGO)
    assert far.in_ego_lane is False


def test_missing_lane_side_means_not_in_lane():
    obj = Obj(track_id=7)
    CollisionEstimator().estimate([obj], ObjectHistoryTracker(), {"left": None, "right": EGO["right"]})
    assert obj.in_ego_lane is False


# ── AlertManager hysteresis ─────────────────────────────────────────────────

def _threat(ttc=1.2):
    o = Obj(track_id=9)
    o.in_ego_lane, o.risk_level, o.ttc_seconds = True, "DANGER", ttc
    return o


def test_alert_needs_persistence_to_trigger_and_to_clear():
    mgr = AlertManager(danger_persist_frames=3, clear_persist_frames=4)
    t = 0.0
    for i in range(2):
        assert mgr.evaluate([_threat()], now=t + i) is None
    alert = mgr.evaluate([_threat()], now=t + 2)
    assert alert is not None and alert.track_id == 9

    for i in range(3):                       # clear frames 1..3: still on
        assert mgr.evaluate([], now=10 + i) is not None
    assert mgr.evaluate([], now=13) is None  # 4th clear frame dismisses


def test_single_frame_spike_does_not_alert():
    mgr = AlertManager(danger_persist_frames=3, clear_persist_frames=4)
    for frame in ([_threat()], [], [_threat()], [], [_threat()]):
        assert mgr.evaluate(frame, now=0.0) is None


def test_alert_duration_uses_frame_time():
    mgr = AlertManager(danger_persist_frames=1, clear_persist_frames=5)
    mgr.evaluate([_threat()], now=100.0)
    alert = mgr.evaluate([_threat()], now=101.5)
    assert alert.seconds_active == pytest.approx(1.5)


def test_most_urgent_threat_is_reported():
    mgr = AlertManager(danger_persist_frames=1)
    slow, fast = _threat(ttc=1.8), _threat(ttc=0.9)
    fast.track_id = 11
    assert mgr.evaluate([slow, fast], now=0.0).track_id == 11


def test_implausible_closing_speed_is_treated_as_noise():
    # 60 m → 40 m in 0.375 s = 53 m/s: distance jitter, not a real threat
    dists = [60.0 - 20.0 * i / 9 for i in range(10)]
    obj = Obj(track_id=12)
    CollisionEstimator(max_closing_speed_mps=40).estimate([obj], _history(12, dists), EGO)
    assert obj.ttc_seconds is None and obj.closing_speed_mps is None
    assert obj.risk_level == "SAFE"


def test_box_extending_below_visible_road_is_clamped_for_lane_test():
    # Lines end at row 400 (hood below). A truck alongside on the left whose
    # box reaches down onto the hood reflection must not count as in-lane.
    ego = {"left": (300, 400, 560, 300), "right": (980, 400, 720, 300)}
    truck = Obj(track_id=13, x1=0, y1=0, x2=520, y2=550)
    CollisionEstimator().estimate([truck], ObjectHistoryTracker(), ego)
    assert truck.in_ego_lane is False
