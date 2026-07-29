"""
schemas.py — Pydantic models defining the exact JSON structure sent per frame
             over the /ws/stream WebSocket endpoint.

Why Pydantic here?
------------------
The WebSocket sends one JSON message per frame. Without a typed schema, that
message is an arbitrary dict — the frontend developer (Day 9) has no guarantee
about what fields exist, what their types are, or when they are None vs absent.
Pydantic solves three problems at once:

  1. **Type safety**: if pipeline code accidentally writes a string where a
     float is expected, Pydantic catches it at serialization time rather than
     causing a silent JSON oddity in the browser.

  2. **Self-documenting contract**: the class definitions below ARE the API
     contract. Anyone reading this file knows exactly what the frontend will
     receive, with no ambiguity.

  3. **FastAPI auto-docs**: FastAPI can expose these models in /docs (Swagger UI)
     so the Day 9 developer can explore the schema without reading source code.

Schema hierarchy (one message per frame):
    FramePayload
    ├── frame_b64          str          # JPEG encoded as base64
    ├── metadata           FrameMetadataSchema
    │   ├── frame_number   int
    │   ├── timestamp      float        # time.perf_counter() at frame capture
    │   ├── fps_current    float
    │   ├── lane_offset    float | None # normalized offset [-1, 1], 0 = centered
    │   ├── active_alert   AlertSchema | None
    │   └── tracked_objects  list[TrackedObjectSchema]
    │       ├── track_id              int
    │       ├── class_name            str
    │       ├── bbox                  list[int]  # [x1, y1, x2, y2]
    │       ├── confidence            float
    │       ├── estimated_distance_m  float | None
    │       ├── closing_speed_mps     float | None
    │       ├── ttc_seconds           float | None
    │       ├── risk_level            str   # "SAFE" | "CAUTION" | "DANGER"
    │       └── in_ego_lane           bool
    └── (AlertSchema — embedded in metadata.active_alert)
        ├── active           bool
        ├── message          str
        ├── severity         str        # "DANGER" | "CAUTION"
        ├── track_id         int | None
        └── duration_seconds float
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Per-object data ──────────────────────────────────────────────────────────

class TrackedObjectSchema(BaseModel):
    """
    Serialized representation of one tracked object for the frontend.

    Maps directly to the TrackedObject dataclass in detection/stage.py,
    with the additional fields added by downstream stages:
      - estimated_distance_m  (DepthEstimationStage)
      - closing_speed_mps     (CollisionFusionStage)
      - ttc_seconds           (CollisionFusionStage)
      - risk_level            (CollisionFusionStage)
      - in_ego_lane           (CollisionFusionStage)
    """

    track_id: int = Field(
        description="ByteTrack persistent ID. Stable across frames for the same physical object."
    )
    class_name: str = Field(
        description="Human-readable COCO class label, e.g. 'car', 'person'."
    )
    bbox: List[int] = Field(
        description="Bounding box as [x1, y1, x2, y2] pixel coordinates (top-left, bottom-right)."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="YOLOv8 detection confidence score in [0.0, 1.0]."
    )

    # Depth stage output (None until DepthEstimationStage has run ≥1 cycle)
    estimated_distance_m: Optional[float] = Field(
        default=None,
        description=(
            "Pseudo-metric distance in metres. "
            "Derived from a calibration scale applied to MiDaS relative depth — "
            "not physically calibrated, but useful for relative ordering. "
            "None if depth has not yet been computed for this object."
        ),
    )

    # Fusion stage output (None until CollisionFusionStage has ≥3 history points)
    closing_speed_mps: Optional[float] = Field(
        default=None,
        description=(
            "Rate of approach in metres/second (positive = closing). "
            "None until at least min_history_points frames of distance data exist."
        ),
    )
    ttc_seconds: Optional[float] = Field(
        default=None,
        description=(
            "Estimated time-to-collision in seconds at current closing speed. "
            "None if closing_speed_mps is None, zero, or negative (retreating)."
        ),
    )
    risk_level: str = Field(
        default="SAFE",
        description="Risk classification: 'SAFE' | 'CAUTION' | 'DANGER'.",
    )
    in_ego_lane: bool = Field(
        default=False,
        description=(
            "True if the object's bounding-box centre falls within the ego lane bounds "
            "computed by LaneDetectionStage. Objects outside the ego lane are "
            "lower priority even at short distances."
        ),
    )


# ── Alert data ───────────────────────────────────────────────────────────────

class AlertSchema(BaseModel):
    """
    Serialized representation of the driver-facing collision alert.

    'active=False' is never sent — if there is no alert the field in
    FrameMetadataSchema is None. This schema only appears when active=True.
    """

    active: bool = Field(
        description="Always True when this schema is present in the payload."
    )
    message: str = Field(
        description="Human-readable warning text, e.g. '⚠ COLLISION RISK — Car #7 | TTC 1.4s'."
    )
    severity: str = Field(
        description="Alert severity: 'DANGER' or 'CAUTION'."
    )
    track_id: Optional[int] = Field(
        default=None,
        description="ByteTrack ID of the highest-priority threatening object.",
    )
    duration_seconds: float = Field(
        description="How long this alert has been continuously active (seconds)."
    )


# ── Per-frame metadata ───────────────────────────────────────────────────────

class FrameMetadataSchema(BaseModel):
    """
    Structured metadata for one processed frame.

    Sent alongside the base64-encoded JPEG in every WebSocket message.
    The frontend uses this to update the dashboard without decoding the image.
    """

    frame_number: int = Field(
        description="Monotonically increasing frame counter (resets to 0 on reconnect)."
    )
    timestamp: float = Field(
        description="time.perf_counter() value at the moment the raw frame was captured."
    )
    fps_current: float = Field(
        description="Smoothed real-time FPS estimate, updated every 15 frames."
    )
    lane_offset: Optional[float] = Field(
        default=None,
        description=(
            "Normalised lateral offset of the ego vehicle within its lane, "
            "ranging from -1.0 (far left) to +1.0 (far right). "
            "0.0 means perfectly centred. None if lane lines were not detected."
        ),
    )
    tracked_objects: List[TrackedObjectSchema] = Field(
        default_factory=list,
        description="All currently tracked objects with their risk annotations.",
    )
    active_alert: Optional[AlertSchema] = Field(
        default=None,
        description="The live driver-facing alert, or null if no alert is active.",
    )


# ── Full WebSocket message ───────────────────────────────────────────────────

class FramePayload(BaseModel):
    """
    The complete message sent to the WebSocket client for every processed frame.

    Why base64-over-JSON instead of a binary WebSocket frame?
    ---------------------------------------------------------
    In a production streaming system you would send the JPEG bytes as a
    WebSocket binary frame and the metadata as a separate JSON message,
    using message ordering or a thin framing protocol to correlate the two.
    That approach is ~33% more bandwidth-efficient (base64 inflates by ~33%).

    For a portfolio project the tradeoffs are different:
      - A single JSON message keeps the frontend code dead simple: one
        `JSON.parse()`, one `img.src = 'data:image/jpeg;base64,' + data.frame_b64`
        — no binary ArrayBuffer handling, no two-message correlation logic.
      - The Day 9 frontend is running locally on the same machine, so bandwidth
        is not a bottleneck.
      - Pydantic can validate and document the entire payload (image + metadata)
        in one schema, which makes the Swagger /docs page far more useful.
      - The ~33% overhead is fully acceptable at ≤30 fps over localhost.
    """

    frame_b64: str = Field(
        description=(
            "The annotated video frame encoded as JPEG and then base64-encoded. "
            "To render in the browser: img.src = 'data:image/jpeg;base64,' + frame_b64"
        )
    )
    metadata: FrameMetadataSchema = Field(
        description="Structured per-frame metadata (detections, lane offset, alert state)."
    )
