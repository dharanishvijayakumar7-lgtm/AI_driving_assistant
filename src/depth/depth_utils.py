"""
depth_utils.py — Utility functions for depth map visualization and
                 per-object distance extraction.

This module contains three key helpers:

1. colorize_depth_map()        — Heatmap visualization of raw depth maps.
2. estimate_object_distance()  — Extract a single depth value for a bounding box.
3. relative_to_pseudo_meters() — Convert relative depth to a rough metric estimate.
"""

from typing import Optional

import cv2
import numpy as np


def colorize_depth_map(
    depth_map: np.ndarray,
    colormap: int = cv2.COLORMAP_INFERNO,
) -> np.ndarray:
    """
    Convert a normalized relative depth map into a colorized BGR heatmap.

    The coloring convention:
      - **Warm colors** (yellow/white) → objects CLOSE to the camera
      - **Cool colors** (dark purple/black) → objects FAR from the camera

    This matches human intuition: "hot" = close/dangerous, "cool" = distant/safe.

    We use COLORMAP_INFERNO (a perceptually uniform sequential colormap from
    matplotlib's "magma" family). It avoids the rainbow artifacts of
    COLORMAP_JET and degrades gracefully for colorblind viewers.

    Args:
        depth_map: Float32 array of shape (H, W) with values in [0, 1].
                   Higher values = closer (MiDaS inverse depth convention).
        colormap:  OpenCV colormap constant. Default: cv2.COLORMAP_INFERNO.

    Returns:
        BGR uint8 image of shape (H, W, 3) suitable for display or overlay.
    """
    # Scale to [0, 255] uint8 for applyColorMap
    depth_uint8 = (depth_map * 255).clip(0, 255).astype(np.uint8)

    # Apply the colormap
    colored = cv2.applyColorMap(depth_uint8, colormap)

    return colored


def estimate_object_distance(
    depth_map: np.ndarray,
    bounding_box: tuple[int, int, int, int],
) -> float:
    """
    Extract a single representative depth value for a tracked object.

    Why MEDIAN instead of MEAN?
    ───────────────────────────
    Bounding boxes are axis-aligned rectangles that inevitably include
    background pixels around the object's actual silhouette (especially for
    non-rectangular shapes like people or motorcycles). These background
    pixels often have very different depth values from the object itself.

    - **Mean** is sensitive to these outlier background pixels. A car at
      depth 0.8 with 30% background at depth 0.2 would report ~0.62 — a
      significant underestimate of closeness.

    - **Median** is robust to outliers. As long as >50% of the box is
      occupied by the object (which is almost always true for a well-fitted
      detection), the median returns a value representative of the actual
      object depth, not the leaked background.

    Additionally, we use the **center crop** (inner 60% of the box) to
    further reduce background contamination at the box edges, where leakage
    is worst.

    Args:
        depth_map:     Float32 array (H, W), normalized [0, 1], higher = closer.
        bounding_box:  (x1, y1, x2, y2) in pixel coordinates.

    Returns:
        A single float in [0, 1] representing the median depth of the object
        region. Higher = object is closer to the camera.
    """
    from src.utils.logger import get_logger as _get_logger
    _log = _get_logger(__name__)

    x1, y1, x2, y2 = bounding_box
    h, w = depth_map.shape[:2]

    # Clamp to frame boundaries
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        # Bug 2 diagnostic: this path means the bbox is degenerate
        # (zero or negative area) after clamping to frame bounds.
        _log.warning(
            "[depth] Degenerate bbox after clamp: (%d,%d,%d,%d) on (%dx%d) map → return 0.0",
            x1, y1, x2, y2, w, h,
        )
        return 0.0

    # Use the inner 60% of the bounding box to reduce edge background leakage
    box_w = x2 - x1
    box_h = y2 - y1
    margin_x = int(box_w * 0.2)
    margin_y = int(box_h * 0.2)

    crop_x1 = x1 + margin_x
    crop_y1 = y1 + margin_y
    crop_x2 = x2 - margin_x
    crop_y2 = y2 - margin_y

    # Fall back to full box if the crop is too small
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        crop_x1, crop_y1, crop_x2, crop_y2 = x1, y1, x2, y2

    roi = depth_map[crop_y1:crop_y2, crop_x1:crop_x2]

    if roi.size == 0:
        _log.warning(
            "[depth] Empty ROI after crop: bbox=(%d,%d,%d,%d) crop=(%d,%d,%d,%d) → return 0.0",
            x1, y1, x2, y2, crop_x1, crop_y1, crop_x2, crop_y2,
        )
        return 0.0

    result = float(np.median(roi))

    # Bug 2 diagnostic: log when the ROI median is near zero (will produce ~200m)
    if result < 0.01:
        _log.warning(
            "[depth] Near-zero median depth %.4f for bbox (%d,%d,%d,%d): "
            "depth map max=%.4f min=%.4f → will clamp to ~200m. "
            "Cause: object bbox may be on a flat/sky region of the depth map.",
            result, x1, y1, x2, y2,
            float(depth_map.max()), float(depth_map.min()),
        )

    return result



def relative_to_pseudo_meters(
    relative_depth: float,
    calibration_scale: float = 30.0,
) -> float:
    """
    Convert a normalized relative depth value to a rough distance in meters.

    ╔══════════════════════════════════════════════════════════════════════╗
    ║  ⚠️  THIS IS AN APPROXIMATION, NOT A METRICALLY ACCURATE DISTANCE. ║
    ╚══════════════════════════════════════════════════════════════════════╝

    Monocular depth models produce relative/ordinal depth — they know "A is
    closer than B" but cannot determine the actual scale (meters, feet, etc.)
    from a single image. True metric depth requires:

      • Stereo cameras: triangulation from two viewpoints with known baseline.
      • LiDAR/radar: direct time-of-flight range measurements.
      • Calibrated mono: known camera intrinsics + ground-plane assumption,
        which is fragile and only works on flat roads.

    Our heuristic:
      pseudo_distance = calibration_scale / (relative_depth + epsilon)

    This assumes inverse depth is roughly proportional to distance (which is
    physically correct for pinhole cameras), then applies a single scale
    factor. The ``calibration_scale`` value should be tuned by eyeballing
    results on your specific camera setup — it is NOT a physically derived
    calibration constant.

    Typical tuning: set calibration_scale so that a car ~10 m ahead of the
    dashcam reads approximately "~10m". Adjust from there.

    Args:
        relative_depth:    Normalized inverse depth in [0, 1] (higher = closer).
        calibration_scale: Scale factor for the heuristic conversion. Default 30.0
                           works reasonably for typical dashcam FOV (~110°).

    Returns:
        Estimated distance in meters (float). Clamped to [0.5, 200.0] to
        avoid degenerate values (div-by-zero → infinity, or negative).
    """
    epsilon = 1e-4  # Prevent division by zero for pixels at "infinite" depth

    pseudo_distance = calibration_scale / (relative_depth + epsilon)

    # Clamp to a sensible range
    return max(0.5, min(200.0, pseudo_distance))
