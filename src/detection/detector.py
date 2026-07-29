"""
detector.py — YOLOv8 wrapper for real-time vehicle and pedestrian detection.

Why YOLOv8n (nano)?
  The nano variant has ~3.2M parameters vs ~68M for YOLOv8x. On CPU it runs
  at ~10–15 FPS vs <1 FPS for the large model. Accuracy is sufficient for the
  driving assistant use-case where we care about bounding box position, not
  fine-grained classification.

Why load the model once in __init__?
  YOLO model initialization involves disk I/O, weight loading into RAM, and a
  JIT warm-up pass. Doing this per-frame would cap us at <1 FPS. Loading once
  and calling the already-warm model each frame is the only way to achieve
  real-time throughput.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from ultralytics import YOLO
import supervision as sv

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# COCO class mapping — only the subset relevant to driving scenes
# ---------------------------------------------------------------------------
# Keys are the string names used in config.yaml; values are the integer COCO
# class IDs that YOLO uses internally. This is the authoritative source of
# truth for which classes the pipeline cares about.
RELEVANT_COCO_CLASSES: dict[str, int] = {
    "person":     0,
    "bicycle":    1,
    "car":        2,
    "motorcycle": 3,
    "bus":        5,
    "truck":      7,
}


@dataclass
class Detection:
    """
    A single detected object in one frame, optionally enriched with a track ID.

    This is the data contract between detector.py, tracker.py, stage.py, and
    any future module that reads meta["tracked_objects"].  Downstream stages
    (depth estimator, fusion engine, alert system) should read from this
    structure rather than parsing raw YOLO output directly.

    Attributes:
        x1, y1, x2, y2: Bounding box corners in pixel coordinates.
        class_id:        COCO integer class ID.
        class_name:      Human-readable class label (e.g., "car").
        confidence:      Model confidence in [0, 1].
        track_id:        Persistent tracker-assigned ID. -1 if not yet tracked.
    """
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float
    track_id: int = -1


class VehicleDetector:
    """
    Wraps a pretrained YOLOv8 model to detect vehicles and pedestrians.

    The detector is stateless between frames — it only produces detections
    for the current frame. Temporal state (tracking IDs) is handled by
    VehicleTracker in tracker.py, keeping concerns cleanly separated.
    """

    def __init__(self, config: dict) -> None:
        """
        Load the YOLO model and prepare class filters.

        Args:
            config: The 'detection' sub-dict from config.yaml. Expected keys:
                    model_path, confidence_threshold, classes, device.

        Raises:
            RuntimeError: If the model fails to load (bad path, corrupt file).
        """
        self._conf_threshold: float = config.get("confidence_threshold", 0.4)
        self._device: str = config.get("device", "cpu")
        model_path: str = config.get("model_path", "yolov8n.pt")
        # imgsz: YOLO internal inference resolution. YOLOv8 letterboxes the
        # input to this square size before running the model. 640 is the
        # training default and gives the best accuracy/speed tradeoff.
        # Reducing to 480 is a ~20% speedup with minimal accuracy impact.
        self._imgsz: int = config.get("imgsz", 640)

        # Map configured class name strings → COCO integer IDs for YOLO's
        # `classes` parameter (which only accepts integer IDs, not strings).
        configured_classes: list[str] = config.get(
            "classes", list(RELEVANT_COCO_CLASSES.keys())
        )
        self._class_ids: list[int] = [
            RELEVANT_COCO_CLASSES[c]
            for c in configured_classes
            if c in RELEVANT_COCO_CLASSES
        ]
        if not self._class_ids:
            raise ValueError(
                "No valid COCO class names found in detection.classes. "
                f"Valid options: {list(RELEVANT_COCO_CLASSES.keys())}"
            )

        logger.info(
            "Loading YOLO model '%s' on device '%s' (conf=%.2f, classes=%s)",
            model_path, self._device, self._conf_threshold, configured_classes,
        )
        try:
            self._model = YOLO(model_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load YOLO model '{model_path}': {exc}") from exc

        # Warm-up pass: run a blank frame through the model so the first real
        # frame isn't penalised by JIT compilation / CUDA kernel init latency.
        logger.info("Warming up YOLO model (one blank inference pass)...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model.predict(dummy, device=self._device, verbose=False)
        logger.info("YOLO model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """
        Run inference on a single frame and return a supervision Detections object.

        The returned object is compatible with VehicleTracker.update() and with
        supervision's annotator helpers. Filtering to only the configured class
        IDs happens inside YOLO (not post-hoc), which is slightly faster because
        the model skips non-target NMS candidates.

        Args:
            frame: A BGR numpy array (H x W x 3) from VideoSource.get_frame().

        Returns:
            sv.Detections with .xyxy, .confidence, .class_id populated.
            .tracker_id is None at this stage — set by VehicleTracker.
        """
        results = self._model.predict(
            frame,
            conf=self._conf_threshold,
            classes=self._class_ids,
            device=self._device,
            imgsz=self._imgsz,
            verbose=False,
        )
        return sv.Detections.from_ultralytics(results[0])

    @property
    def class_names(self) -> dict[int, str]:
        """
        Map from COCO integer class ID to human-readable class name.

        Sourced directly from the YOLO model so it's always consistent with
        whatever weights file was loaded.
        """
        return self._model.names  # type: ignore[return-value]
