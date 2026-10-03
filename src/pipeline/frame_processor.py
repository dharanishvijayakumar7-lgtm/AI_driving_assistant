"""
frame_processor.py — Pluggable per-frame processing pipeline.

Why a stage-list pattern?
  The main loop doesn't need to know *what* processing happens to each frame —
  it just calls `processor.process(frame)` and gets back the result. New
  capabilities (object detection, lane detection, depth estimation, etc.) are
  added by registering a new Stage without touching main.py or video_source.py.

  Each Stage is a callable that receives:
      (frame: np.ndarray, meta: dict) -> (frame: np.ndarray, meta: dict)

  The `meta` dict is a shared scratchpad that stages can write to (e.g., a
  detector writes bounding boxes into meta["detections"] and a later fusion
  stage reads them). This avoids coupling stages to each other directly.

How to extend on Day 2:
    from src.pipeline.frame_processor import FrameProcessor, Stage

    class MyDetector:
        def __call__(self, frame, meta):
            # ... run model, populate meta["detections"] ...
            return frame, meta

    processor = FrameProcessor()
    processor.add_stage("detector", MyDetector())
"""

from typing import Any, Callable, Optional

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type alias for a processing stage.
# Signature: (frame, metadata_dict) -> (frame, metadata_dict)
Stage = Callable[
    [np.ndarray, dict[str, Any]],
    tuple[np.ndarray, dict[str, Any]],
]


class FrameProcessor:
    """
    Orchestrates an ordered list of processing stages applied to each frame.

    Stages are plain callables (functions or objects implementing __call__)
    that transform a frame and/or annotate the shared metadata dict.  Because
    stages are just callables registered by name, they can be added, removed,
    or reordered at runtime without any changes to the calling code.

    Stages registered by main.py / the WebSocket handler, in order:
      - "resize"    : FrameResizeStage — downscale to the working resolution
      - "detection" : YOLOv8 detection + ByteTrack tracking
      - "lanes"     : Classical lane detection + ego-lane corridor
      - "depth"     : MiDaS monocular depth → per-object distance
      - "fusion"    : Closing speed, TTC and collision risk
      - "alerts"    : Debounced driver alert + final label layer
    """

    def __init__(self) -> None:
        # Ordered list of (name, stage) pairs.
        # A list (not a dict) preserves insertion order for deterministic execution.
        self._stages: list[tuple[str, Stage]] = []
        # Per-stage timing accumulators, logged every _timing_interval frames
        self._timing_accum: dict[str, float] = {}
        self._timing_frame_count: int = 0
        self._timing_interval: int = 30
        logger.debug("FrameProcessor initialized with no stages.")

    def add_stage(self, name: str, stage: Stage) -> None:
        """
        Append a processing stage to the end of the pipeline.

        Args:
            name:  A human-readable identifier (e.g., "detector").
                   Used in log messages and for future stage lookup/removal.
            stage: A callable with signature
                   ``(frame: np.ndarray, meta: dict) -> (np.ndarray, dict)``.

        Example:
            processor.add_stage("detector", YOLODetector(model_path="..."))
        """
        self._stages.append((name, stage))
        logger.info("Stage '%s' added to FrameProcessor (position %d).", name, len(self._stages))

    def remove_stage(self, name: str) -> bool:
        """
        Remove the first stage with the given name.

        Args:
            name: The name used when the stage was added.

        Returns:
            True if a stage was found and removed; False otherwise.
        """
        for i, (stage_name, _) in enumerate(self._stages):
            if stage_name == name:
                del self._stages[i]
                logger.info("Stage '%s' removed from FrameProcessor.", name)
                return True
        logger.warning("Stage '%s' not found; nothing removed.", name)
        return False

    def process(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Run the frame through every registered stage in order.

        Each stage receives the *output* of the previous stage, so they form
        a true pipeline. The shared ``meta`` dict accumulates annotations
        (detections, lane lines, depth maps, risk scores) as stages execute.

        Timing: every 30 frames, logs per-stage millisecond breakdown at INFO
        level so the FPS bottleneck is immediately visible.

        Args:
            frame:     The raw BGR frame from VideoSource.get_frame().
            timestamp: Capture time of the frame in seconds (pass
                       VideoSource.timestamp). Stored as meta["timestamp"] so
                       time-based stages (closing speed, TTC) measure video
                       time rather than processing time. Defaults to now.

        Returns:
            A tuple of:
              - The processed frame (may have overlays drawn on it).
              - The final metadata dict containing all stage outputs.
        """
        import time as _time

        # Stages draw their overlays in place on `frame`, so analysis stages
        # must not read it: MiDaS would estimate depth on painted boxes and
        # the lane fill, and Canny would find edges on box outlines. They read
        # meta["clean_frame"] instead — an un-annotated frame that is never
        # drawn on. FrameResizeStage replaces it with the resized frame.
        meta: dict[str, Any] = {
            "timestamp": timestamp if timestamp is not None else _time.perf_counter(),
            "clean_frame": frame,
        }
        first_stage = self._stages[0][1] if self._stages else None
        if not getattr(first_stage, "provides_clean_frame", False):
            frame = frame.copy()   # separate drawing canvas

        for name, stage in self._stages:
            try:
                t0 = _time.perf_counter()
                frame, meta = stage(frame, meta)
                elapsed_ms = (_time.perf_counter() - t0) * 1000
                self._timing_accum[name] = self._timing_accum.get(name, 0.0) + elapsed_ms
            except Exception as exc:  # noqa: BLE001
                # Log and skip a failing stage rather than crashing the pipeline.
                logger.error("Stage '%s' raised an exception and was skipped: %s", name, exc)

        self._timing_frame_count += 1

        if self._timing_frame_count % self._timing_interval == 0:
            n = self._timing_interval
            parts = []
            total = 0.0
            for sname, _ in self._stages:
                avg = self._timing_accum.get(sname, 0.0) / n
                total += avg
                parts.append(f"{sname}={avg:.1f}ms")
            parts.append(f"TOTAL={total:.1f}ms")
            logger.info("[Pipeline timing] last %d frames — %s", n, "  ".join(parts))
            self._timing_accum.clear()

        return frame, meta

    @property
    def stage_names(self) -> list[str]:
        """Return the ordered list of registered stage names (read-only view)."""
        return [name for name, _ in self._stages]
