"""
collision_estimator.py — TTC computation + risk classification.

For each tracked object with sufficient distance history, this module:

  1. Computes **closing speed** via linear least-squares regression across
     the recent distance-vs-time history.
  2. Derives **time-to-collision (TTC)** = current_distance / closing_speed.
  3. Classifies **risk level** (SAFE / CAUTION / DANGER) using configurable
     TTC thresholds.
  4. Determines **in_ego_lane** by checking whether the object's horizontal
     center falls between the detected lane lines.

Why linear fit across N frames instead of two-frame differencing?
-----------------------------------------------------------------
Depth estimation from monocular models (Day 4, MiDaS) produces noisy per-frame
distance values. Two consecutive frames might yield distances of 18.3 m and
17.9 m — a 0.4 m drop in ~33 ms, implying a closing speed of 12 m/s — or 17.9 m
and 18.5 m — suggesting the object is *retreating* at 18 m/s. Both are noise.

A **least-squares linear fit across 5-10 frames** acts as a principled smoother:

  - It minimizes the sum of squared residuals, so individual noisy readings
    have limited influence on the estimated slope.
  - The slope represents the *average trend* over the window, which is far
    more stable than any single Δd/Δt measurement.
  - No additional hyperparameter tuning is needed (unlike exponential moving
    averages, which require a decay factor).
  - The fit can be computed in O(N) with the closed-form normal equation —
    no iterative optimization, negligible compute cost.

This is the same principle behind Kalman filters' smoothing effect, but
simpler and fully explainable: "I take the best-fit line through the last
N distance measurements and read off its slope."

Why we guard against non-positive closing speed
-------------------------------------------------
TTC = distance / closing_speed. If closing_speed ≤ 0, the object is either:

  - **Stationary** (closing_speed ≈ 0): TTC would be ±infinity or a
    divide-by-zero. No collision risk from a stationary relative distance.
  - **Moving away** (closing_speed < 0): TTC would be negative, which is
    physically meaningless — you can't collide with something that's
    getting farther away.

In both cases we report TTC = None and risk = SAFE.

Lane-relevance filtering
--------------------------
An object in a DANGER TTC state that is NOT in the ego vehicle's lane
presents a very different threat level than one directly ahead. A car three
lanes over closing at 5 m/s is of little concern to the ego vehicle — it's
likely on a different trajectory entirely (passing, merging elsewhere, etc.).

We use a simple heuristic: if both ego lane lines are detected, check whether
the object's horizontal center falls between them. If it does, the object is
``in_ego_lane = True`` and retains its full risk level. If not, we downgrade
the risk by one level (DANGER → CAUTION, CAUTION → SAFE) to avoid false
alarm fatigue.

This is NOT precise multi-lane geometry — we don't model curves, lane widths,
or adjacent lane assignments. It's a pragmatic filter that significantly
reduces false DANGER alerts from objects that are clearly off to the side.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from src.fusion.object_history import DistanceObservation, ObjectHistoryTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Risk level constants ──────────────────────────────────────────────────────
RISK_DANGER = "DANGER"
RISK_CAUTION = "CAUTION"
RISK_SAFE = "SAFE"


@dataclass
class CollisionRisk:
    """
    The collision-risk annotation attached to each tracked object.

    Attributes
    ----------
    closing_speed_mps : float or None
        Rate at which the object is approaching, in meters per second.
        Positive = closing in, negative = moving away.
        None if insufficient history to estimate.
    ttc_seconds : float or None
        Time-to-collision in seconds. None if the object is not approaching
        or if there is insufficient data.
    risk_level : str
        One of "DANGER", "CAUTION", "SAFE".
    in_ego_lane : bool
        True if the object's horizontal center falls between the detected
        ego lane lines. False if it doesn't or if lane lines are unavailable.
    """

    closing_speed_mps: Optional[float]
    ttc_seconds: Optional[float]
    risk_level: str
    in_ego_lane: bool


class CollisionEstimator:
    """
    Estimates closing speed, TTC, and collision risk for tracked objects.

    Parameters
    ----------
    ttc_danger_threshold : float
        TTC (seconds) below which risk is classified as DANGER.
    ttc_caution_threshold : float
        TTC (seconds) below which risk is CAUTION (if above danger threshold).
    min_history_points : int
        Minimum number of distance observations required before computing
        a TTC estimate. Avoids noisy estimates from 1–2 data points.
    """

    def __init__(
        self,
        ttc_danger_threshold: float = 2.0,
        ttc_caution_threshold: float = 4.0,
        min_history_points: int = 3,
    ) -> None:
        self._ttc_danger = ttc_danger_threshold
        self._ttc_caution = ttc_caution_threshold
        self._min_points = min_history_points

        logger.info(
            "CollisionEstimator ready (danger<%.1fs, caution<%.1fs, min_points=%d).",
            self._ttc_danger,
            self._ttc_caution,
            self._min_points,
        )

    def estimate(
        self,
        tracked_objects: list,
        object_history: ObjectHistoryTracker,
        lane_lines: Optional[dict[str, Any]] = None,
    ) -> list:
        """
        For each tracked object, compute closing speed, TTC, risk level,
        and ego-lane membership. Results are attached directly to each object
        as dynamic attributes.

        Parameters
        ----------
        tracked_objects : list[TrackedObject]
            From DetectionStage (meta["tracked_objects"]).
        object_history : ObjectHistoryTracker
            The rolling distance history buffer.
        lane_lines : dict or None
            From LaneDetectionStage: {"left": (x1,y1,x2,y2)|None,
                                       "right": (x1,y1,x2,y2)|None}.

        Returns
        -------
        list[TrackedObject]
            Same objects, now enriched with ``closing_speed_mps``,
            ``ttc_seconds``, ``risk_level``, ``in_ego_lane`` attributes.
        """
        for obj in tracked_objects:
            history = object_history.get_history(obj.track_id)

            # Compute closing speed and TTC
            risk = self._compute_risk(obj, history)

            # Determine ego-lane membership
            in_ego_lane = self._is_in_ego_lane(obj, lane_lines)
            risk = CollisionRisk(
                closing_speed_mps=risk.closing_speed_mps,
                ttc_seconds=risk.ttc_seconds,
                risk_level=risk.risk_level,
                in_ego_lane=in_ego_lane,
            )

            # Lane-relevance filtering: downgrade risk for objects NOT in ego lane.
            # A fast-closing object three lanes over is much less concerning than
            # one directly ahead, so we reduce alert fatigue by lowering its risk.
            if not in_ego_lane:
                risk = self._downgrade_risk(risk)

            # Attach results as dynamic attributes on the TrackedObject
            obj.closing_speed_mps = risk.closing_speed_mps
            obj.ttc_seconds = risk.ttc_seconds
            obj.risk_level = risk.risk_level
            obj.in_ego_lane = risk.in_ego_lane

        return tracked_objects

    # ── Internal computation ─────────────────────────────────────────────

    def _compute_risk(
        self,
        obj: Any,
        history: Optional[deque[DistanceObservation]],
    ) -> CollisionRisk:
        """
        Compute closing speed, TTC, and raw risk level from distance history.
        """
        # Not enough data yet — can't estimate anything reliably
        if history is None or len(history) < self._min_points:
            return CollisionRisk(
                closing_speed_mps=None,
                ttc_seconds=None,
                risk_level=RISK_SAFE,
                in_ego_lane=False,
            )

        # ── Linear least-squares fit: distance = a*t + b ──────────────
        # We want the slope `a` of the best-fit line through
        # (timestamp, distance) pairs. A negative slope means distance
        # is decreasing over time → the object is getting closer.
        #
        # Closed-form: slope = (n*Σ(ti*di) - Σti*Σdi) / (n*Σ(ti²) - (Σti)²)
        #
        # We shift timestamps to start at 0 for numerical stability
        # (avoids large floating-point values in the products).

        t0 = history[0].timestamp
        times = np.array([obs.timestamp - t0 for obs in history], dtype=np.float64)
        dists = np.array([obs.distance_m for obs in history], dtype=np.float64)

        n = len(times)
        sum_t = np.sum(times)
        sum_d = np.sum(dists)
        sum_tt = np.sum(times * times)
        sum_td = np.sum(times * dists)

        denominator = n * sum_tt - sum_t * sum_t

        if abs(denominator) < 1e-12:
            # All timestamps are identical (shouldn't happen in practice) —
            # can't compute a slope.
            return CollisionRisk(
                closing_speed_mps=None,
                ttc_seconds=None,
                risk_level=RISK_SAFE,
                in_ego_lane=False,
            )

        slope = (n * sum_td - sum_t * sum_d) / denominator

        # slope is dd/dt (meters per second).
        # Negative slope = distance decreasing = object approaching.
        # We define closing_speed as POSITIVE when closing in,
        # so: closing_speed = -slope.
        closing_speed = -slope

        # Current distance is the most recent observation
        current_distance = dists[-1]

        # ── TTC computation ──────────────────────────────────────────────
        # TTC = current_distance / closing_speed
        # ONLY valid when closing_speed > 0 (object is approaching).
        # Guard against non-positive closing speed to avoid:
        #   - Division by zero (stationary object)
        #   - Negative TTC (object moving away) — physically meaningless
        if closing_speed > 0.1:  # small threshold to filter near-zero noise
            ttc = current_distance / closing_speed
        else:
            ttc = None

        # ── Risk classification ──────────────────────────────────────────
        if ttc is not None and ttc < self._ttc_danger:
            risk_level = RISK_DANGER
        elif ttc is not None and ttc < self._ttc_caution:
            risk_level = RISK_CAUTION
        else:
            risk_level = RISK_SAFE

        return CollisionRisk(
            closing_speed_mps=round(closing_speed, 2),
            ttc_seconds=round(ttc, 1) if ttc is not None else None,
            risk_level=risk_level,
            in_ego_lane=False,  # will be set by caller
        )

    @staticmethod
    def _is_in_ego_lane(obj: Any, lane_lines: Optional[dict]) -> bool:
        """
        Check if the object's horizontal center is between the left and right
        ego lane lines.

        We evaluate the lane line positions at the object's vertical center
        (y-coordinate) by linear interpolation along each lane line segment.
        This handles the fact that lane lines converge toward the horizon —
        checking at the object's y-level gives a much more accurate answer
        than checking at a fixed y.

        Returns False if either lane line is undetected (we can't determine
        lane membership with only one boundary).
        """
        if lane_lines is None:
            return False

        left_line = lane_lines.get("left")
        right_line = lane_lines.get("right")

        # Need both lane lines to determine ego-lane membership
        if left_line is None or right_line is None:
            return False

        # Object horizontal center
        obj_cx = (obj.x1 + obj.x2) / 2.0
        # Object vertical center — evaluate lane lines at this y
        obj_cy = (obj.y1 + obj.y2) / 2.0

        # Interpolate each lane line's x-position at obj_cy
        left_x = _interpolate_lane_x(left_line, obj_cy)
        right_x = _interpolate_lane_x(right_line, obj_cy)

        if left_x is None or right_x is None:
            return False

        # Ensure left < right (lane lines might be stored in any order)
        lo, hi = min(left_x, right_x), max(left_x, right_x)

        return lo <= obj_cx <= hi

    @staticmethod
    def _downgrade_risk(risk: CollisionRisk) -> CollisionRisk:
        """
        Reduce risk by one level for objects outside the ego lane.

        DANGER  → CAUTION
        CAUTION → SAFE
        SAFE    → SAFE  (no change)
        """
        downgraded_level = risk.risk_level
        if risk.risk_level == RISK_DANGER:
            downgraded_level = RISK_CAUTION
        elif risk.risk_level == RISK_CAUTION:
            downgraded_level = RISK_SAFE

        return CollisionRisk(
            closing_speed_mps=risk.closing_speed_mps,
            ttc_seconds=risk.ttc_seconds,
            risk_level=downgraded_level,
            in_ego_lane=risk.in_ego_lane,
        )


def _interpolate_lane_x(
    line: tuple[int, int, int, int], target_y: float
) -> Optional[float]:
    """
    Given a lane line segment (x1, y1, x2, y2), interpolate the x position
    at *target_y* using linear interpolation.

    Returns None if the line is horizontal (Δy ≈ 0) or if target_y is
    outside the line segment's y-range (extrapolation is unreliable for
    short Hough line segments).
    """
    x1, y1, x2, y2 = line

    dy = y2 - y1
    if abs(dy) < 1e-6:
        return None  # horizontal line — can't interpolate in y

    # Allow a small margin outside the segment for practical robustness
    y_min = min(y1, y2)
    y_max = max(y1, y2)

    # Clamp target_y to the line segment's range instead of returning None.
    # This handles objects that are slightly above/below the visible lane
    # line segment — the lane boundary continues beyond what Hough detected,
    # and clamping gives a reasonable approximation.
    clamped_y = max(y_min, min(y_max, target_y))

    t = (clamped_y - y1) / dy
    return x1 + t * (x2 - x1)
