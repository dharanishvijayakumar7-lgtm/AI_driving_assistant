"""
stage.py — LaneDetectionStage: FrameProcessor-compatible adapter for lane detection.

Follows the same (frame, meta) → (frame, meta) contract as Day 2's DetectionStage,
so it plugs into the pipeline with a single processor.add_stage() call in main.py.
"""

from typing import Any

import numpy as np

from src.lanes.lane_detector import LaneDetector
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LaneDetectionStage:
    """
    Wraps LaneDetector into the FrameProcessor stage interface.

    After this stage executes, the metadata dict carries:
      meta["lane_lines"]  = {"left": (x1,y1,x2,y2) | None,
                              "right": (x1,y1,x2,y2) | None}
      meta["lane_offset"] = {"pixels": float,
                              "normalized": float}   # [-1,1], + = right drift

    Future stages (fusion engine, alert system) read from these keys to
    determine risk without re-running lane detection.
    """

    def __init__(self, config: dict) -> None:
        self._detector = LaneDetector(config)
        logger.info("LaneDetectionStage ready.")

    def __call__(
        self, frame: np.ndarray, meta: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        result = self._detector.detect(frame)

        meta["lane_lines"] = {
            "left":  result.left_line,
            "right": result.right_line,
        }
        meta["lane_offset"] = {
            "pixels":     result.lane_offset_pixels,
            "normalized": result.lane_offset_normalized,
        }

        # Draw overlay on frame (import here to avoid circular imports at module load)
        from src.visualization.display import draw_lane_overlay
        frame = draw_lane_overlay(frame, result)

        return frame, meta
