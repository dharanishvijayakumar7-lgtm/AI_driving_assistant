"""
websocket_handler.py — WebSocket endpoint /ws/stream.

Runs the full Days 1-6 pipeline internally and streams two things per frame:
  1. The annotated frame encoded as JPEG → base64 (for the img tag).
  2. A FrameMetadataSchema JSON blob (for the dashboard widgets).

Both are packed into a single FramePayload JSON message so the frontend
needs exactly one `JSON.parse()` call and one `img.src = ...` assignment.

Design notes
------------
- The pipeline is initialised fresh per WebSocket connection (not shared). This
  means each client gets its own VideoSource, so multiple simultaneous clients
  each replay the video from the start. For Day 9 (single-client demo) this is
  perfectly fine, and it avoids thread-safety headaches around shared state.

- The main loop is an async generator driven by `asyncio.to_thread()`. The
  heavy CPU work (YOLO, MiDaS, OpenCV blending) runs in a threadpool worker,
  keeping the asyncio event loop free to service other HTTP routes (like /health)
  and WebSocket control messages (ping/pong, disconnect).

- On client disconnect `websocket.send_text()` raises WebSocketDisconnect, which
  is caught here and triggers a clean pipeline teardown (source.release()).

- JPEG quality is tunable. 85 is a good default: it gives visually lossless
  frames at roughly 15–40 KB each, which at 10-15 fps = 150–600 KB/s over
  localhost — well within any practical limit.
"""

from __future__ import annotations

import asyncio
import base64
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from src.api.schemas import AlertSchema, FrameMetadataSchema, FramePayload, TrackedObjectSchema
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Path helpers — resolve project root from this file's location
# src/api/websocket_handler.py → up two levels → project root
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent


# ── JPEG encoding quality (0–100). ────────────────────────────────────────────
# 65 for WebSocket streaming: ~40% smaller payload than 85, imperceptible
# quality difference at video frame rates (8-15 fps). Use 85+ for offline
# recording or when saving individual frames for analysis.
# Bug 3 fix: lowering quality is the correct tradeoff for a streaming
# pipeline; note this in comments + README rather than hiding it.
_JPEG_QUALITY = 65


# ── Public endpoint function ──────────────────────────────────────────────────

async def stream_pipeline(websocket: WebSocket) -> None:
    """
    WebSocket endpoint handler for ``/ws/stream``.

    Lifecycle:
      1. Accept the connection.
      2. Build the full pipeline (VideoSource + FrameProcessor + all stages)
         using the same config.yaml as main.py.
      3. Enter the async frame loop:
           a. Read next frame (blocking I/O → run in threadpool).
           b. Process frame through all stages (CPU-heavy → run in threadpool).
           c. Encode annotated frame as JPEG → base64 (threadpool).
           d. Extract metadata from frame_meta dict.
           e. Send FramePayload JSON to client.
      4. On stream end or client disconnect: release VideoSource and log summary.

    Parameters
    ----------
    websocket : WebSocket
        The connected WebSocket client (injected by FastAPI).
    """
    await websocket.accept()
    logger.info("WebSocket client connected: %s", websocket.client)

    source = None
    frame_count = 0
    fps_display = 0.0
    loop_start = time.perf_counter()
    fps_timer_start = loop_start

    try:
        # ── 1. Build pipeline ─────────────────────────────────────────────
        source, processor, source_meta = await asyncio.to_thread(_build_pipeline)
        logger.info("Pipeline built for WebSocket client. Source: %s", source_meta)

        disp_cfg = _load_config().get("display", {})
        target_w: Optional[int] = disp_cfg.get("width")
        target_h: Optional[int] = disp_cfg.get("height")

        # ── 2. Frame loop ─────────────────────────────────────────────────
        # Timing accumulators for the breakdown log (Bug 3 instrumentation)
        t_pipeline_total = 0.0
        t_encode_total   = 0.0
        t_send_total     = 0.0

        while True:
            # ── FPS estimate (update every 5 frames for responsiveness) ───
            if frame_count > 0 and frame_count % 5 == 0:
                elapsed = time.perf_counter() - fps_timer_start
                fps_display = 5.0 / elapsed if elapsed > 0 else 0.0
                fps_timer_start = time.perf_counter()

            # ── Process + encode in ONE threadpool call ────────────────────
            # Bug 3 fix: was TWO asyncio.to_thread() calls (pipeline + encode).
            # Each to_thread() incurs scheduler overhead on Windows (~1-3 ms).
            # Merging into one call eliminates the per-frame second hop.
            result = await asyncio.to_thread(
                _process_and_encode,
                source, processor, target_w, target_h,
                frame_count + 1, fps_display,
            )

            if result is None:
                # Stream ended (video file exhausted or webcam disconnected)
                logger.info("Video stream ended. Closing WebSocket.")
                break

            payload_json, pipeline_ms, encode_ms = result
            frame_count += 1
            t_pipeline_total += pipeline_ms
            t_encode_total   += encode_ms

            # ── Send to client ─────────────────────────────────────────────
            t_send_start = time.perf_counter()
            await websocket.send_text(payload_json)
            send_ms = (time.perf_counter() - t_send_start) * 1000
            t_send_total += send_ms

            # ── Timing breakdown log every 30 frames ───────────────────────
            if frame_count % 30 == 0:
                n = 30
                logger.info(
                    "[WS timing] last %d frames — "
                    "pipeline=%.1fms  encode=%.1fms  send=%.1fms  "
                    "total=%.1fms  fps=%.1f",
                    n,
                    t_pipeline_total / n,
                    t_encode_total   / n,
                    t_send_total     / n,
                    (t_pipeline_total + t_encode_total + t_send_total) / n,
                    fps_display,
                )
                t_pipeline_total = t_encode_total = t_send_total = 0.0

    except WebSocketDisconnect:
        logger.info(
            "WebSocket client disconnected after %d frames (%.1f s).",
            frame_count,
            time.perf_counter() - loop_start,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error in WebSocket stream: %s", exc)
        try:
            await websocket.close(code=1011)  # Internal Error
        except Exception:  # noqa: BLE001
            pass
    finally:
        if source is not None:
            source.release()
        logger.info(
            "WebSocket pipeline torn down. Total frames: %d, avg FPS: %.1f",
            frame_count,
            frame_count / max(time.perf_counter() - loop_start, 0.001),
        )


# ── Pipeline construction (runs in threadpool) ────────────────────────────────

def _load_config() -> dict:
    """Load and return the project config. Called once per connection."""
    from src.utils.config import load_config
    config_path = _PROJECT_ROOT / "configs" / "config.yaml"
    return load_config(str(config_path))


def _build_pipeline():
    """
    Construct VideoSource + FrameProcessor with all registered stages.

    This is a direct mirror of the setup logic in main.py — deliberately kept
    in sync with it rather than abstracted, so main.py remains the authoritative
    reference and both entry points stay easy to read and compare.

    Returns
    -------
    (VideoSource, FrameProcessor, VideoMetadata)
    """
    from src.pipeline.video_source import VideoSource
    from src.pipeline.frame_processor import FrameProcessor

    config = _load_config()

    # ── Video source ──────────────────────────────────────────────────────
    src_cfg = config["source"]
    source = VideoSource(
        source_type=src_cfg["type"],
        file_path=src_cfg.get("file_path"),
        webcam_index=src_cfg.get("webcam_index", 0),
    )

    # ── Processor stages (same order as main.py — ORDER MATTERS) ─────────
    processor = FrameProcessor()

    # ── Stage 0: pre-resize (Day 8 addition — mirrors main.py) ───────────
    pipeline_cfg = config.get("pipeline", {})
    resize_w = pipeline_cfg.get("resize_width", 1280)
    resize_h = pipeline_cfg.get("resize_height", 720)
    if resize_w and resize_h:
        from src.pipeline.resize_stage import FrameResizeStage
        processor.add_stage("resize", FrameResizeStage(width=resize_w, height=resize_h))

    if "detection" in config:
        from src.detection.stage import DetectionStage
        processor.add_stage("detection", DetectionStage(config["detection"]))

    if "lanes" in config:
        from src.lanes.stage import LaneDetectionStage
        lanes_cfg = dict(config["lanes"])
        lanes_cfg.setdefault(
            "show_lane_overlay",
            config.get("visualization", {}).get("show_lane_overlay", True),
        )
        processor.add_stage("lanes", LaneDetectionStage(lanes_cfg))

    if "depth" in config:
        from src.depth.stage import DepthEstimationStage
        depth_cfg = dict(config["depth"])
        depth_cfg["show_heatmap_overlay"] = config.get("visualization", {}).get(
            "show_depth_panel", depth_cfg.get("show_heatmap_overlay", True)
        )
        processor.add_stage("depth", DepthEstimationStage(depth_cfg))

    if "fusion" in config:
        from src.fusion.stage import CollisionFusionStage
        processor.add_stage("fusion", CollisionFusionStage(config["fusion"]))

    if "alerts" in config:
        from src.alerts.stage import AlertStage
        processor.add_stage(
            "alerts",
            AlertStage(
                config=config["alerts"],
                viz_config=config.get("visualization", {}),
            ),
        )

    logger.info(
        "WebSocket pipeline ready: %d stages: %s",
        len(processor.stage_names),
        processor.stage_names,
    )
    return source, processor, source.metadata


# ── Single-frame processing + encoding (runs in ONE threadpool call) ─────────

def _process_and_encode(
    source,
    processor,
    target_w: Optional[int],
    target_h: Optional[int],
    frame_number: int,
    fps: float,
) -> Optional[tuple[str, float, float]]:
    """
    Read + process one frame through the full pipeline, then JPEG-encode
    and serialise to JSON — all in a single threadpool call.

    Bug 3 fix rationale:
      Previously the loop called asyncio.to_thread() TWICE per frame:
        1. _process_one_frame()  — YOLO + MiDaS + all stages  (~110 ms)
        2. _build_payload_json() — JPEG encode + base64 + JSON (~10-20 ms)
      Each to_thread() incurs Windows scheduler overhead of ~1-3 ms.
      At 8 FPS that's up to 48 ms/frame of pure thread-hop tax.
      Merging into one call eliminates the second to_thread() entirely.

    Also fixes: HUD FPS was always 0.0 because fps wasn't available when
    _process_one_frame ran. Now fps is passed in and drawn correctly.

    Returns
    -------
    (payload_json, pipeline_ms, encode_ms) tuple, or None if source exhausted.
    """
    import time as _time

    # ─ Pipeline ─────────────────────────────────────────────────────────────
    t_pipe = _time.perf_counter()
    frame = source.get_frame()
    if frame is None:
        return None

    annotated_frame, frame_meta = processor.process(frame)

    if target_w and target_h:
        h, w = annotated_frame.shape[:2]
        if w != target_w or h != target_h:
            annotated_frame = cv2.resize(
                annotated_frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )

    from src.visualization.display import draw_hud
    draw_hud(annotated_frame, fps=fps, meta=frame_meta)  # real fps, not 0.0
    pipeline_ms = (_time.perf_counter() - t_pipe) * 1000

    # ─ Encode ────────────────────────────────────────────────────────────────
    t_enc = _time.perf_counter()
    payload_json = _build_payload_json(annotated_frame, frame_meta, frame_number, fps)
    encode_ms = (_time.perf_counter() - t_enc) * 1000

    return payload_json, pipeline_ms, encode_ms


# ── Payload serialisation (runs in threadpool) ────────────────────────────────

def _build_payload_json(
    frame: np.ndarray,
    frame_meta: dict[str, Any],
    frame_number: int,
    fps: float,
) -> str:
    """
    Convert an annotated BGR frame + pipeline metadata into a JSON string.

    Steps:
      1. JPEG-encode the frame (cv2.imencode — fast C++ path).
      2. base64-encode the JPEG bytes.
         WHY base64-over-JSON instead of binary WebSocket frames:
           - One JSON.parse() on the frontend vs. binary frame + separate JSON
             correlation logic — dramatically simpler Day 9 code.
           - No ArrayBuffer handling in JavaScript.
           - Pydantic can document the entire payload (image + metadata) together.
           - Bandwidth penalty (~33% larger) is irrelevant over localhost at
             10-15 fps for a portfolio project. A production system would use
             binary frames + a framing protocol.
      3. Extract typed metadata from the raw frame_meta dict.
      4. Assemble FramePayload and serialise to JSON string.

    Parameters
    ----------
    frame        : Annotated BGR frame (NumPy ndarray).
    frame_meta   : Dict written by the pipeline stages.
    frame_number : Monotonically increasing frame index.
    fps          : Current smoothed FPS estimate.

    Returns
    -------
    str — JSON string ready to pass to websocket.send_text().
    """
    # ── 1+2. JPEG → base64 ────────────────────────────────────────────────
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, _JPEG_QUALITY]
    success, jpeg_buf = cv2.imencode(".jpg", frame, encode_params)
    if not success:
        raise RuntimeError("cv2.imencode failed — frame encoding error.")
    frame_b64 = base64.b64encode(jpeg_buf.tobytes()).decode("ascii")

    # ── 3. Extract tracked objects ────────────────────────────────────────
    raw_objects: list = frame_meta.get("tracked_objects", [])
    tracked_schemas: list[TrackedObjectSchema] = []

    for obj in raw_objects:
        tracked_schemas.append(
            TrackedObjectSchema(
                track_id=getattr(obj, "track_id", -1),
                class_name=getattr(obj, "class_name", "unknown"),
                bbox=[
                    int(getattr(obj, "x1", 0)),
                    int(getattr(obj, "y1", 0)),
                    int(getattr(obj, "x2", 0)),
                    int(getattr(obj, "y2", 0)),
                ],
                confidence=float(getattr(obj, "confidence", 0.0)),
                estimated_distance_m=_safe_float(getattr(obj, "estimated_distance_m", None)),
                closing_speed_mps=_safe_float(getattr(obj, "closing_speed_mps", None)),
                ttc_seconds=_safe_float(getattr(obj, "ttc_seconds", None)),
                risk_level=str(getattr(obj, "risk_level", "SAFE")),
                in_ego_lane=bool(getattr(obj, "in_ego_lane", False)),
            )
        )

    # ── 4. Extract alert ──────────────────────────────────────────────────
    raw_alert = frame_meta.get("active_alert")
    alert_schema: Optional[AlertSchema] = None
    if raw_alert is not None:
        alert_schema = AlertSchema(
            active=True,
            message=str(getattr(raw_alert, "message", "")),
            severity=str(getattr(raw_alert, "severity", "DANGER")),
            track_id=_safe_int(getattr(raw_alert, "track_id", None)),
            duration_seconds=float(getattr(raw_alert, "seconds_active", 0.0)),
        )

    # ── 5. Extract lane offset ────────────────────────────────────────────
    lane_info = frame_meta.get("lane_offset", {})
    lane_offset: Optional[float] = None
    if lane_info and isinstance(lane_info, dict):
        lane_offset = _safe_float(lane_info.get("normalized"))

    # ── 6. Assemble payload ───────────────────────────────────────────────
    metadata = FrameMetadataSchema(
        frame_number=frame_number,
        timestamp=time.perf_counter(),
        fps_current=round(fps, 2),
        lane_offset=lane_offset,
        tracked_objects=tracked_schemas,
        active_alert=alert_schema,
    )

    payload = FramePayload(frame_b64=frame_b64, metadata=metadata)

    # model_dump_json() is Pydantic v2's fast JSON serialiser (Rust-backed).
    # It produces a compact JSON string without extra whitespace.
    return payload.model_dump_json()


# ── Utility helpers ───────────────────────────────────────────────────────────

def _safe_float(value: Any) -> Optional[float]:
    """Return float(value) or None if value is None / not convertible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Return int(value) or None if value is None / not convertible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
