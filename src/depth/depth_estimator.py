"""
depth_estimator.py — Wraps a pretrained monocular depth model for per-frame
                     relative depth estimation.

Model choice: MiDaS (via torch.hub) vs. Depth Anything (via HuggingFace)
─────────────────────────────────────────────────────────────────────────
We choose **MiDaS** for the following reasons:

1. Zero extra dependencies — MiDaS loads through `torch.hub`, and we already
   have `torch` installed from Day 2 (YOLOv8). Depth Anything requires the
   `transformers` library (~200 MB + tokenizers), adding significant install
   size for marginal quality gain at the "small" tier.

2. Simpler integration — MiDaS provides its own preprocessing transforms via
   `torch.hub.load(... , "transforms")`. Depth Anything through HuggingFace
   requires a separate `AutoImageProcessor` that adds another abstraction layer.

3. Proven track record — MiDaS v3.1 (DPT-based) is battle-tested in dozens
   of monocular depth projects and has well-documented behavior for the small
   and hybrid model variants we need for real-time inference.

Trade-off: Depth Anything (v2, 2024) can produce slightly sharper depth edges,
especially around thin structures like poles. If we later need that quality
jump, swapping is straightforward — this class isolates all model details.

Why monocular depth is RELATIVE, not metric
───────────────────────────────────────────
A single 2D image lacks the parallax information that stereo cameras or the
time-of-flight measurements that LiDAR provide. The model learns *ordinal*
depth relationships ("A is closer than B") from training data, but it cannot
determine absolute scale from one image alone. The output is a dense map where
higher values = closer and lower values = farther (for MiDaS), but the
absolute numbers are arbitrary and shift between frames.

To get real-world metric depth you need:
  • A calibrated stereo camera pair (depth from disparity)
  • LiDAR / radar range measurements (direct time-of-flight)
  • A calibrated monocular setup with known camera intrinsics + ground plane
    assumptions (fragile, works only on flat roads)

We apply a configurable `calibration_scale` heuristic in depth_utils.py to
produce pseudo-metric estimates for downstream display purposes, but these
are explicitly approximate.
"""

from typing import Optional

import cv2
import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)

# MiDaS model variants available via torch.hub
_MIDAS_MODELS: dict[str, str] = {
    "midas_small":  "MiDaS_small",       # Fastest, ~2 MB, good for real-time
    "midas_hybrid": "DPT_Hybrid",        # Better quality, ~120 MB, slower
    "midas_large":  "DPT_Large",         # Best quality, ~1.2 GB, slow on CPU
}


class DepthEstimator:
    """
    Wraps a MiDaS monocular depth model.

    The model is loaded once at initialization. Each call to ``estimate()``
    runs inference on a single frame and returns a per-pixel relative depth
    map (numpy array, same H×W as the input, float32).

    Depth values represent **relative inverse depth** (higher = closer),
    NOT real-world meters. This is an inherent limitation of monocular depth
    estimation — see the module docstring above for a full explanation.

    Args:
        config: The ``depth`` sub-dict from config.yaml. Expected keys:
            - model_name:       One of "midas_small", "midas_hybrid", "midas_large"
            - input_resolution: Internal inference size (int). The model resizes
                                frames to this resolution internally. Smaller =
                                faster but coarser depth maps. 256 is a reasonable
                                default for real-time on CPU.
            - device:           "auto", "cpu", or "cuda" (inherited from detection config
                                or overridden per-section).

    Usage:
        estimator = DepthEstimator(config["depth"])
        depth_map = estimator.estimate(bgr_frame)
        # depth_map.shape == (frame_height, frame_width), dtype=float32
    """

    def __init__(self, config: dict) -> None:
        self._model_name = config.get("model_name", "midas_small")
        self._input_resolution = config.get("input_resolution", 256)

        # ── Device resolution ────────────────────────────────────────────
        device_str: str = config.get("device", "auto")
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device_str)
        logger.info("DepthEstimator device: %s", self._device)

        # ── Load MiDaS model via torch.hub ───────────────────────────────
        hub_name = _MIDAS_MODELS.get(self._model_name)
        if hub_name is None:
            raise ValueError(
                f"Unknown depth model '{self._model_name}'. "
                f"Choose from: {list(_MIDAS_MODELS.keys())}"
            )

        logger.info(
            "Loading MiDaS model '%s' (hub name: '%s') — "
            "this may download weights on first run...",
            self._model_name, hub_name,
        )

        self._model = torch.hub.load(
            "intel-isl/MiDaS", hub_name, trust_repo=True
        )
        self._model.to(self._device)
        self._model.eval()

        # ── Load matching transforms ─────────────────────────────────────
        # MiDaS provides purpose-built transforms per model variant.
        midas_transforms = torch.hub.load(
            "intel-isl/MiDaS", "transforms", trust_repo=True
        )
        if self._model_name == "midas_small":
            self._transform = midas_transforms.small_transform
        elif self._model_name == "midas_hybrid":
            self._transform = midas_transforms.dpt_transform
        else:  # midas_large
            self._transform = midas_transforms.dpt_transform

        logger.info(
            "DepthEstimator ready (model=%s, input_res=%d, device=%s).",
            self._model_name, self._input_resolution, self._device,
        )

    @torch.no_grad()
    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Produce a per-pixel relative depth map from a single BGR frame.

        The returned depth map has the same (H, W) spatial dimensions as the
        input ``frame``. Values are float32 representing **relative inverse
        depth**: higher values indicate pixels that are *closer* to the camera.

        The absolute magnitude of the values is meaningless across frames —
        they are internally normalized to [0, 1] within each frame for
        consistency, but this does NOT correspond to any physical distance.

        Args:
            frame: Input frame in BGR format (H, W, 3), uint8.

        Returns:
            depth_map: np.ndarray of shape (H, W), dtype float32, values in
                       [0.0, 1.0] where 1.0 = closest, 0.0 = farthest.
        """
        h, w = frame.shape[:2]

        # MiDaS expects RGB input
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply the model-specific preprocessing transform
        input_batch = self._transform(rgb).to(self._device)

        # Run inference
        prediction = self._model(input_batch)

        # Resize prediction to original frame size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_map = prediction.cpu().numpy()

        # Normalize to [0, 1] — MiDaS outputs inverse depth (higher = closer),
        # so we keep that convention after normalization.
        d_min = depth_map.min()
        d_max = depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        return depth_map.astype(np.float32)
