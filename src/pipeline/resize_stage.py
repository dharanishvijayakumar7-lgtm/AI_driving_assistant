"""
resize_stage.py — FrameResizeStage: downsizes every frame once before any
                  other stage sees it, eliminating the 4K processing tax.

Why a dedicated first stage instead of resizing inside each stage?
------------------------------------------------------------------
Every compute-heavy stage (YOLO, Canny+Hough, MiDaS preprocessing) paid the
cost of operating on the full 4K source frame (3840×2160 = 8.3 MP). Canny
alone on 8MP was the single biggest bottleneck at ~180 ms / frame.

A single cv2.resize call at the pipeline entry point:
  1. Costs ~2–5 ms (trivial compared to what it saves).
  2. Immediately makes ALL downstream stages faster — no per-stage changes.
  3. Means all overlay coordinates (bounding boxes, lane lines, depth PIP)
     are in the resized coordinate space, so they're consistent and don't
     need scaling later.
  4. Fixes the lane detection tuning mismatch: the Hough thresholds
     (hough_min_line_length=25, threshold=15) were calibrated for ~720p frames,
     not 4K.  At 4K, a 25-pixel segment is <0.7% of frame width — it picks up
     texture noise.  At 720p, 25px ≈ 2% width, which is the intended scale.
  5. The display.width/height resize in show_frame() becomes a no-op (frame is
     already the target size), saving a second redundant resize per frame.

Config (add to config.yaml under a new 'pipeline:' section):
    pipeline:
      resize_width: 1280
      resize_height: 720
Set either to null to keep the original source resolution.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FrameResizeStage:
    """
    Stage 0 of the pipeline: resize the incoming frame to a smaller resolution.

    Placing this first means every subsequent stage (YOLO, Canny+Hough,
    MiDaS, drawing calls) operates on the smaller frame. The display window
    already specifies its own target size via display.width/height in
    config.yaml, so no coordinate remapping is needed downstream.

    Args:
        width:  Target width in pixels. None = keep source width.
        height: Target height in pixels. None = keep source height.

    Example (main.py):
        processor.add_stage("resize", FrameResizeStage(width=1280, height=720))
    """

    def __init__(
        self,
        width: Optional[int] = 1280,
        height: Optional[int] = 720,
    ) -> None:
        self._width = width
        self._height = height
        if width and height:
            logger.info(
                "FrameResizeStage: will resize every frame to %dx%d before processing. "
                "This fixes lane-detection threshold calibration and speeds up all downstream stages.",
                width, height,
            )
        else:
            logger.info("FrameResizeStage: no resize configured — pass-through mode.")

    def __call__(
        self, frame: np.ndarray, meta: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if self._width is None or self._height is None:
            return frame, meta

        h, w = frame.shape[:2]
        if w == self._width and h == self._height:
            return frame, meta   # already the right size — skip the copy

        # INTER_LINEAR is the best tradeoff: faster than INTER_CUBIC, sharper
        # than INTER_NEAREST. For downscaling video frames it is standard.
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_LINEAR
        )
        return frame, meta
