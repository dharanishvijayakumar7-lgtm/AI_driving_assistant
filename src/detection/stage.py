"""
stage.py — DetectionStage: the FrameProcessor-compatible wrapper that
           combines VehicleDetector + VehicleTracker into one pipeline stage.

Why a separate stage.py instead of putting this logic in detector.py?
  detector.py and tracker.py are pure algorithmic components — they know
  nothing about the pipeline they live in.  stage.py is the adapter that
  translates between those components and the FrameProcessor's
  (frame, meta) contract.  This separation keeps each file focused and
  makes it easy to test the detector and tracker in isolation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.detection.detector import VehicleDetector, Detection
from src.detection.tracker import VehicleTracker
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedObject:
    """
    The data contract that this stage writes into meta["tracked_objects"].

    Future stages (lane detector, depth estimator, fusion engine, alert system)
    MUST read from this dataclass rather than raw YOLO/supervision internals.
    This decouples downstream stages from the specific detection library.

    Attributes:
        x1, y1, x2, y2: Bounding box in pixel coordinates (top-left, bottom-right).
        class_name:      Human-readable class label, e.g. "car", "person".
        confidence:      Detection confidence in [0.0, 1.0].
        track_id:        ByteTrack persistent ID. Stable across frames for the
                         same physical object. -1 if tracking failed.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    class_name: str
    confidence: float
    track_id: int


class DetectionStage:
    """
    FrameProcessor stage that runs detection + tracking on every frame.

    This is the only class that main.py needs to import from the detection
    package. It implements the stage callable interface:

        (frame: np.ndarray, meta: dict) -> (frame: np.ndarray, meta: dict)

    After this stage runs, meta["tracked_objects"] contains a list of
    TrackedObject instances that every subsequent stage can read.

    How it hooks into FrameProcessor (the one-line change in main.py):
        processor.add_stage("detection", DetectionStage(config["detection"]))
    """

    def __init__(self, config: dict) -> None:
        """
        Resolve the device, then initialize detector and tracker.

        Args:
            config: The 'detection' sub-dict from config.yaml.

        Side-effects:
            Downloads yolov8n.pt from Ultralytics CDN if not already cached
            (happens once, then stored in the ultralytics cache directory).
        """
        # Resolve device: "auto" → probe torch for CUDA, fall back to CPU.
        device: str = config.get("device", "auto")
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
            logger.info("Device auto-detected: '%s'", device)
        else:
            logger.info("Device set by config: '%s'", device)

        resolved_config = {**config, "device": device}

        self._detector = VehicleDetector(resolved_config)
        self._tracker = VehicleTracker(resolved_config)
        logger.info("DetectionStage ready.")

    def __call__(
        self, frame: np.ndarray, meta: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Run detection + tracking and annotate the frame with bounding boxes.

        Pipeline:
            1. VehicleDetector.detect()  → sv.Detections (class, box, conf)
            2. VehicleTracker.update()   → sv.Detections + tracker_id
            3. Convert → list[TrackedObject] → meta["tracked_objects"]
            4. draw_detections()         → annotated frame (in-place)

        Args:
            frame: Raw BGR frame from VideoSource.
            meta:  Shared metadata dict (may contain data from earlier stages).

        Returns:
            (annotated_frame, meta) — meta now includes "tracked_objects".
        """
        # 1. Detect
        sv_detections = self._detector.detect(frame)

        # 2. Track
        sv_tracked = self._tracker.update(sv_detections)

        # 3. Convert supervision Detections → TrackedObject list
        tracked_objects: list[TrackedObject] = []
        class_names = self._detector.class_names

        if len(sv_tracked) > 0:
            for i in range(len(sv_tracked)):
                x1, y1, x2, y2 = sv_tracked.xyxy[i].astype(int)
                class_id = int(sv_tracked.class_id[i])
                conf = float(sv_tracked.confidence[i])
                track_id = (
                    int(sv_tracked.tracker_id[i])
                    if sv_tracked.tracker_id is not None
                    else -1
                )
                name = class_names.get(class_id, f"class_{class_id}")
                tracked_objects.append(
                    TrackedObject(
                        x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                        class_name=name,
                        confidence=conf,
                        track_id=track_id,
                    )
                )

        meta["tracked_objects"] = tracked_objects

        # 4. Annotate frame (import here to avoid circular imports)
        from src.visualization.display import draw_detections
        frame = draw_detections(frame, tracked_objects)

        logger.debug(
            "DetectionStage: %d objects tracked this frame.", len(tracked_objects)
        )
        return frame, meta
