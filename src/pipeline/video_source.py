"""
video_source.py — Unified video input abstraction.

Why this exists:
  Whether the pipeline reads from a video file or a live webcam, the rest of
  the system should not care — it just calls `get_frame()` and receives a
  NumPy array (or None when the stream ends). This class encapsulates all
  OpenCV capture logic so no other module ever touches `cv2.VideoCapture`.
"""

from typing import Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class VideoMetadata:
    """
    Snapshot of the video source properties, captured at open-time.

    Attributes:
        fps:         Native frames-per-second of the source.
        width:       Native frame width in pixels.
        height:      Native frame height in pixels.
        frame_count: Total number of frames for file sources; -1 for webcams.
        source_type: "file" or "webcam" — informational only.
    """

    def __init__(
        self,
        fps: float,
        width: int,
        height: int,
        frame_count: int,
        source_type: str,
    ) -> None:
        self.fps = fps
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self.source_type = source_type

    def __repr__(self) -> str:
        return (
            f"VideoMetadata(source={self.source_type}, "
            f"{self.width}x{self.height} @ {self.fps:.2f}fps, "
            f"frames={self.frame_count})"
        )


class VideoSource:
    """
    Wraps cv2.VideoCapture to provide a clean, unified interface for both
    video files and live webcam feeds.

    Design rationale:
      - Initialization validates the source immediately so startup failures
        are caught before the main loop begins.
      - `get_frame()` returns None (instead of raising) when the stream is
        exhausted, letting the caller decide how to handle end-of-stream.
      - Metadata is read once at open-time and cached so callers don't pay
        repeated cv2 property lookups.

    Usage:
        source = VideoSource(source_type="file", file_path="data/sample.mp4")
        while (frame := source.get_frame()) is not None:
            process(frame)
        source.release()
    """

    def __init__(
        self,
        source_type: str,
        file_path: Optional[str] = None,
        webcam_index: int = 0,
    ) -> None:
        """
        Open the video source and read its metadata.

        Args:
            source_type:  "file" or "webcam".
            file_path:    Path to the video file (required when source_type=="file").
            webcam_index: Device index for the webcam (required when source_type=="webcam").

        Raises:
            ValueError:      If source_type is unrecognized or required args are missing.
            FileNotFoundError: If source_type is "file" and the path does not exist.
            RuntimeError:    If OpenCV cannot open the requested source.
        """
        self._cap: Optional[cv2.VideoCapture] = None
        self._source_type = source_type

        if source_type == "file":
            if not file_path:
                raise ValueError(
                    "file_path must be provided when source_type is 'file'."
                )
            import os
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Video file not found: '{file_path}'. "
                    "Check the file_path setting in configs/config.yaml."
                )
            self._cap = cv2.VideoCapture(file_path)
            logger.info("Opening video file: '%s'", file_path)

        elif source_type == "webcam":
            self._cap = cv2.VideoCapture(webcam_index)
            logger.info("Opening webcam at device index: %d", webcam_index)

        else:
            raise ValueError(
                f"Unknown source_type '{source_type}'. Must be 'file' or 'webcam'."
            )

        if not self._cap.isOpened():
            source_desc = (
                f"file '{file_path}'"
                if source_type == "file"
                else f"webcam index {webcam_index}"
            )
            raise RuntimeError(
                f"OpenCV could not open {source_desc}. "
                "For webcams, verify the device is connected and not in use by another app."
            )

        self._metadata = self._read_metadata()
        logger.info("Video source ready: %s", self._metadata)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_frame(self) -> Optional[np.ndarray]:
        """
        Read and return the next frame from the source.

        Returns:
            A BGR NumPy array (H x W x 3) if a frame was successfully read,
            or None if the stream has ended (EOF for files, capture error for webcams).
        """
        if self._cap is None:
            return None

        ret, frame = self._cap.read()
        if not ret:
            if self._source_type == "file":
                logger.info("End of video file reached.")
            else:
                logger.warning(
                    "Failed to read frame from webcam. "
                    "The device may have been disconnected."
                )
            return None

        return frame

    @property
    def metadata(self) -> VideoMetadata:
        """Return the cached metadata snapshot for this source."""
        return self._metadata

    def release(self) -> None:
        """
        Release the underlying cv2.VideoCapture handle.

        Always call this when done — even if the stream ended naturally —
        to free OS-level resources (file handles, camera drivers).
        """
        if self._cap and self._cap.isOpened():
            self._cap.release()
            logger.info("Video source released.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _read_metadata(self) -> VideoMetadata:
        """
        Query cv2 for source properties and return a VideoMetadata instance.

        For webcams, frame_count is set to -1 because it is undefined.
        """
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = (
            int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if self._source_type == "file"
            else -1
        )
        return VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            frame_count=frame_count,
            source_type=self._source_type,
        )
