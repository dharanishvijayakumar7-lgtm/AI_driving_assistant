"""
object_history.py — Per-track distance/time history buffer.

Maintains a short rolling window of (timestamp, estimated_distance_m) pairs
for every active track ID. This is the temporal memory that makes closing-speed
estimation possible — without it, each frame would be an isolated snapshot
with no concept of "how fast is this object getting closer?".

Design decisions
----------------
* **Rolling window, not unbounded list**: We keep at most `max_length` entries
  per track. This caps memory at O(max_tracks × max_length) regardless of
  video duration. For a 30-fps, 1-hour video that's 108 000 frames — without
  the cap, memory would grow linearly.

* **Timeout-based expiration**: When a tracked object leaves the frame, its
  ByteTrack ID stops receiving updates. After `timeout_seconds` with no new
  observation, we purge the history. This prevents stale IDs from accumulating
  (ByteTrack may reuse IDs in rare cases, and old data from a different
  physical object would corrupt the closing-speed estimate).

* **collections.deque**: O(1) append and O(1) popleft for the rolling window,
  versus O(n) for list slicing. The `maxlen` parameter handles eviction
  automatically.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DistanceObservation:
    """A single (timestamp, distance) measurement for one tracked object."""

    timestamp: float  # seconds (time.perf_counter or frame-derived)
    distance_m: float  # estimated distance from depth stage


class ObjectHistoryTracker:
    """
    Maintains per-track rolling histories of distance observations.

    Usage::

        tracker = ObjectHistoryTracker(max_length=10, timeout_seconds=2.0)

        # Every frame:
        tracker.update(track_id=7, distance_m=18.3, timestamp=t)

        # Query:
        history = tracker.get_history(track_id=7)
        # → deque([DistanceObservation(t0, d0), ..., DistanceObservation(tn, dn)])

        # Periodically:
        tracker.expire_stale(current_time=t)

    Parameters
    ----------
    max_length : int
        Maximum number of observations to keep per track. Older entries are
        silently evicted when this limit is reached. Recommended: 10–20 frames
        (at 30 fps that's ~0.3–0.7 s of history, enough for a stable linear fit
        without introducing lag from stale data).
    timeout_seconds : float
        If a track hasn't been updated for this many seconds, its history is
        eligible for expiration. This prevents ghost entries from objects that
        left the frame.
    """

    def __init__(
        self,
        max_length: int = 10,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._max_length = max_length
        self._timeout = timeout_seconds

        # track_id → deque of DistanceObservation (newest at the right end)
        self._histories: dict[int, deque[DistanceObservation]] = {}

        # track_id → timestamp of the last update (for expiration)
        self._last_seen: dict[int, float] = {}

        logger.info(
            "ObjectHistoryTracker ready (max_length=%d, timeout=%.1fs).",
            max_length,
            timeout_seconds,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def update(self, track_id: int, distance_m: float, timestamp: float) -> None:
        """
        Record a new distance observation for *track_id*.

        If this is the first observation for the track, a new history buffer is
        created. If the buffer is already at ``max_length``, the oldest entry
        is silently evicted.

        Parameters
        ----------
        track_id : int
            ByteTrack persistent ID from DetectionStage.
        distance_m : float
            Estimated distance in (pseudo) meters from DepthEstimationStage.
        timestamp : float
            Current time in seconds (e.g. ``time.perf_counter()``).
        """
        if track_id not in self._histories:
            self._histories[track_id] = deque(maxlen=self._max_length)

        self._histories[track_id].append(
            DistanceObservation(timestamp=timestamp, distance_m=distance_m)
        )
        self._last_seen[track_id] = timestamp

    def get_history(self, track_id: int) -> Optional[deque[DistanceObservation]]:
        """
        Return the rolling distance history for *track_id*, or ``None`` if no
        history exists (track was never seen or has expired).
        """
        return self._histories.get(track_id)

    def expire_stale(self, current_time: float) -> int:
        """
        Remove histories for tracks that haven't been updated recently.

        Parameters
        ----------
        current_time : float
            The current timestamp. Any track whose last update is older than
            ``current_time - timeout_seconds`` is purged.

        Returns
        -------
        int
            Number of tracks expired in this call.
        """
        expired_ids = [
            tid
            for tid, last_t in self._last_seen.items()
            if (current_time - last_t) > self._timeout
        ]
        for tid in expired_ids:
            del self._histories[tid]
            del self._last_seen[tid]

        if expired_ids:
            logger.debug("Expired %d stale track histories.", len(expired_ids))

        return len(expired_ids)

    @property
    def active_track_count(self) -> int:
        """Number of tracks currently held in memory."""
        return len(self._histories)
