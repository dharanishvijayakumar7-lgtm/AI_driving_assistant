"""
tracker.py — ByteTrack multi-object tracking wrapper.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
HOW BYTETRACK WORKS (the algorithm, not just the API)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

The core insight: traditional trackers (SORT, DeepSORT) only match
high-confidence detections, which means they lose tracks whenever
a detector's confidence briefly dips (e.g. object partially occluded).
ByteTrack uses ALL detections — both high and low confidence — in a
two-step matching strategy:

Step 1 — High-confidence matching:
  Take every detection above `high_conf_det_threshold` (~0.6).
  For each existing active track, predict where it should be now
  using a Kalman Filter (models constant-velocity motion). Then
  compute IoU (Intersection over Union) between each predicted
  track box and each high-confidence detection box. Use the Hungarian
  algorithm to find the globally optimal assignment. Matched pairs
  get their track updated; matched detection confirms the track is
  still alive.

Step 2 — Low-confidence rescue:
  Detections that scored BELOW the high-conf threshold but ABOVE the
  activation threshold are now matched against the tracks LEFT
  UNMATCHED in Step 1. Why? Because a slightly occluded car might
  score 0.35 confidence — not enough to start a new track, but enough
  to confirm an existing one hasn't actually disappeared. This is the
  "byte" in ByteTrack.

Lifecycle management:
  - A new detection not matched to any track starts as a "tentative"
    track and is only confirmed after `minimum_consecutive_frames`
    consecutive matches.
  - A confirmed track that goes unmatched for `lost_track_buffer` frames
    is marked "lost" and eventually deleted. During the lost window,
    the Kalman filter keeps extrapolating its position, so it can be
    re-matched if the object reappears.

Result: IDs stay stable through brief occlusions (a car going behind a
lamppost), low-light frames (confidence drops), and camera motion —
exactly the conditions that break SORT-style trackers.

Library note:
  We use `trackers.ByteTrackTracker` (the `trackers` package by Roboflow),
  which is the successor to `supervision.ByteTrack` (deprecated in sv 0.28).
  Its `update(sv.Detections)` method is directly compatible with Ultralytics
  YOLO output converted via `sv.Detections.from_ultralytics()`.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import warnings
import supervision as sv
from trackers import ByteTrackTracker

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VehicleTracker:
    """
    Wraps trackers.ByteTrackTracker to assign persistent IDs to detections.

    This class is intentionally stateful — it must persist across frames so
    the internal Kalman filters and track lifecycle state aren't reset.
    Instantiate once in DetectionStage.__init__(), not per-frame.

    Input:  sv.Detections (output of VehicleDetector.detect())
    Output: sv.Detections with .tracker_id populated for each object.
    """

    def __init__(self, config: dict) -> None:
        """
        Initialize ByteTrackTracker with parameters from config.

        Args:
            config: The 'detection' sub-dict from config.yaml. Reads the
                    nested 'tracker' sub-dict for these keys:
                    - track_activation_threshold (default 0.25)
                    - lost_track_buffer          (default 30 frames)
                    - minimum_matching_threshold  (default 0.1 IoU)
                    - frame_rate                 (default 30 fps)
        """
        tracker_cfg: dict = config.get("tracker", {})

        track_activation_threshold: float = tracker_cfg.get(
            "track_activation_threshold", 0.25
        )
        lost_track_buffer: int = tracker_cfg.get("lost_track_buffer", 30)
        minimum_matching_threshold: float = tracker_cfg.get(
            "minimum_matching_threshold", 0.1
        )
        frame_rate: int = tracker_cfg.get("frame_rate", 30)

        # Suppress the trackers library's own FutureWarnings about internal
        # target= defaults that are unrelated to our usage.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = ByteTrackTracker(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_iou_threshold=minimum_matching_threshold,
                frame_rate=float(frame_rate),
            )

        logger.info(
            "ByteTrackTracker initialized (activation_thresh=%.2f, lost_buffer=%d, "
            "iou_thresh=%.2f, frame_rate=%d)",
            track_activation_threshold,
            lost_track_buffer,
            minimum_matching_threshold,
            frame_rate,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """
        Run ByteTrack's two-step matching and return detections with track IDs.

        Args:
            detections: Output of VehicleDetector.detect() for the current frame.

        Returns:
            sv.Detections with .tracker_id array filled in.
            Detections that could not be matched to any track are dropped.
        """
        if len(detections) == 0:
            return detections

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                tracked = self._tracker.update(detections)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ByteTrack update failed (%s); returning untracked detections.", exc)
            return detections

        logger.debug(
            "Tracker: %d detections in → %d tracks out.",
            len(detections),
            len(tracked),
        )
        return tracked

    def reset(self) -> None:
        """
        Reset all track state.

        Useful when switching video source mid-run to prevent stale Kalman
        states from corrupting new-scene tracks.
        """
        self._tracker.reset()
        logger.info("ByteTrackTracker state reset.")
