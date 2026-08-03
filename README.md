# AI Driving Assistant 🚗

A real-time computer vision pipeline for autonomous driving assistance,
built from scratch in 10 days. The system detects road objects, tracks them
across frames, estimates depth, computes time-to-collision, raises debounced
collision alerts, and streams everything — annotated frames + structured JSON
metadata — over WebSocket to a React dashboard.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Input Source                             │
│            VideoSource (file .mp4 / webcam)                     │
└───────────────────────────┬─────────────────────────────────────┘
                            │ raw BGR frame
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 0 — FrameResizeStage                                     │
│  Resizes every frame to 1280×720 before any inference.          │
│  Effect: lane detection 9.8× faster, overall pipeline 3.2×      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ 1280×720 frame
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1 — DetectionStage                                        │
│  YOLOv8n inference (imgsz=640) → ByteTrack multi-object         │
│  tracking → meta["tracked_objects"] list                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │ + tracked_objects
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2 — LaneDetectionStage                                    │
│  Canny edge detection + ROI trapezoid + HoughLinesP →           │
│  left/right lane lines + normalized ego-lane offset             │
│  meta["lane_lines"], meta["lane_offset"]                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ + lane_lines, lane_offset
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3 — DepthEstimationStage                                  │
│  MiDaS small (every 3rd frame, cached) → relative depth map →   │
│  per-object pseudo-metric distance (obj.estimated_distance_m)   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ + depth per object
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4 — CollisionFusionStage                                  │
│  Rolling distance buffer (10 frames) → linear least-squares     │
│  closing speed → TTC → SAFE/CAUTION/DANGER risk level          │
│  + in_ego_lane filter (horizontal centre vs. lane boundaries)   │
└───────────────────────────┬─────────────────────────────────────┘
                            │ + risk_level, ttc_seconds per object
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5 — AlertStage                                            │
│  Debounced hysteresis (5 frames trigger, 10 frames clear) →     │
│  meta["active_alert"] + baked visual banner on frame            │
└───────────────────────────┬─────────────────────────────────────┘
                            │ fully annotated frame + meta dict
                 ┌──────────┴───────────────┐
                 ▼                          ▼
     ┌─────────────────┐        ┌──────────────────────┐
     │   OpenCV window │        │  FastAPI WebSocket    │
     │   (main.py)     │        │  /ws/stream           │
     └─────────────────┘        │  JPEG base64 frame +  │
                                │  FramePayload JSON    │
                                └──────────┬───────────┘
                                           │
                                           ▼
                                ┌──────────────────────┐
                                │   React Dashboard    │
                                │  (frontend/  port    │
                                │   5173)              │
                                │                      │
                                │  VideoFeed  (img)    │
                                │  AlertBanner (meta)  │
                                │  ObjectPanel (meta)  │
                                │  StatsBar    (meta)  │
                                └──────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Object Detection | [YOLOv8n](https://docs.ultralytics.com/) (Ultralytics) |
| Multi-Object Tracking | [ByteTrack](https://github.com/ifzhang/ByteTrack) via supervision |
| Lane Detection | Classical CV — Canny + HoughLinesP (OpenCV) |
| Depth Estimation | [MiDaS small](https://github.com/isl-org/MiDaS) via `torch.hub` |
| Temporal Fusion | Custom linear least-squares TTC estimator |
| Alert System | Two-counter debounce/hysteresis state machine |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) + WebSocket stream |
| Frontend | [React 18](https://react.dev/) + [Vite 5](https://vitejs.dev/) |
| Config | YAML (`configs/config.yaml`) — zero hardcoded values in `src/` |

---

## Project Structure

```
ai-driving-assistant/
├── src/
│   ├── main.py                    # Entry point — OpenCV window mode
│   ├── run_server.py              # FastAPI server launcher
│   ├── api/
│   │   ├── app.py                 # FastAPI app + CORS + /health
│   │   ├── websocket_handler.py   # /ws/stream — full pipeline over WebSocket
│   │   └── schemas.py             # Pydantic models (FramePayload, etc.)
│   ├── pipeline/
│   │   ├── video_source.py        # Webcam / video file abstraction
│   │   ├── frame_processor.py     # Pluggable stage-list engine
│   │   └── resize_stage.py        # FrameResizeStage (pre-pipeline resize)
│   ├── detection/
│   │   ├── detector.py            # YOLOv8 wrapper
│   │   ├── tracker.py             # ByteTrack wrapper
│   │   └── stage.py               # DetectionStage pipeline adapter
│   ├── lanes/
│   │   ├── lane_detector.py       # Classical lane detection
│   │   ├── lane_utils.py          # Lane math helpers
│   │   └── stage.py               # LaneDetectionStage adapter
│   ├── depth/
│   │   ├── depth_estimator.py     # MiDaS wrapper
│   │   ├── depth_utils.py         # Colormap, distance extraction
│   │   └── stage.py               # DepthEstimationStage adapter
│   ├── fusion/
│   │   ├── object_history.py      # Per-track distance rolling buffer
│   │   ├── collision_estimator.py # TTC computation + risk classification
│   │   └── stage.py               # CollisionFusionStage adapter
│   ├── alerts/
│   │   ├── alert_manager.py       # Priority, debounce & hysteresis
│   │   ├── sound_alert.py         # Optional audio beep
│   │   └── stage.py               # AlertStage adapter
│   ├── utils/
│   │   ├── config.py              # YAML config loader
│   │   └── logger.py              # Centralized logging
│   └── visualization/
│       └── display.py             # HUD, FPS overlay, risk-coded boxes
├── frontend/                      # React dashboard (Day 10)
│   ├── src/
│   │   ├── App.jsx                # Root layout component
│   │   ├── index.jsx              # React entry point
│   │   ├── index.css              # Design system (dark HUD aesthetic)
│   │   ├── components/
│   │   │   ├── VideoFeed.jsx      # Live frame display
│   │   │   ├── AlertBanner.jsx    # React-driven alert from metadata
│   │   │   ├── ObjectPanel.jsx    # Tracked object side panel
│   │   │   └── StatsBar.jsx       # FPS, lane offset, connection
│   │   └── hooks/
│   │       └── useWebSocketStream.js  # WebSocket + reconnect logic
│   ├── package.json
│   └── vite.config.js
├── configs/
│   └── config.yaml                # All runtime settings
├── models/                        # Downloaded .pt weights (gitignored)
├── data/
│   └── sample_videos/             # Drop .mp4 clips here
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Python ≥ 3.9
- Node.js ≥ 18

### 1. Install Python dependencies

> ⚠️ Install PyTorch CPU build **first** to avoid downloading the 2.5 GB CUDA build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### 2. Add a test video

Drop any `.mp4` driving clip into `data/sample_videos/` and set `config.yaml`:

```yaml
source:
  type: "file"
  file_path: "data/sample_videos/your_clip.mp4"
```

### 3. Run: OpenCV window mode (no frontend needed)

```bash
python src/main.py
```

Press **`q`** to quit. YOLO and MiDaS weights auto-download on first run.

### 4. Run: WebSocket server mode + React dashboard

**Terminal 1 — Start the backend:**

```bash
python src/run_server.py
# FastAPI starts on http://localhost:8000
# WebSocket stream: ws://localhost:8000/ws/stream
# Swagger docs:     http://localhost:8000/docs
```

**Terminal 2 — Start the frontend:**

```bash
cd frontend
npm install       # first time only
npm run dev
# Dashboard:  http://localhost:5173
```

Open `http://localhost:5173` — the annotated video feed and live metadata
panels appear automatically.

### 5. Webcam mode

```yaml
source:
  type: "webcam"
  webcam_index: 0
```

No code changes needed.

---

## Performance Benchmarks

All measurements on **CPU-only** hardware (Intel CPU, no GPU).
Source: 3840×2160 video → 1280×720 pipeline working resolution.

### Per-Stage Timing (after Day 8 optimizations)

| Stage | Time (ms) | Notes |
|-------|-----------|-------|
| Resize (4K→720p) | 6.3 | New stage 0 — trivial cost |
| Detection (YOLO) | 55.8 | YOLOv8n, imgsz=640 |
| Lane Detection | 18.6 | 9.8× faster after resize fix |
| Depth (MiDaS) | 28.5 | every 3rd frame cached |
| Fusion (TTC) | 0.3 | pure math, negligible |
| Alerts | 4.3 | 6.6× faster after resize |
| **Total** | **113.7** | |
| **FPS** | **~8.8** | |

> **Context:** 8-9 FPS CPU-only with YOLO + MiDaS + all overlays is at or
> above the practical ceiling for a full CV pipeline on CPU.
> On an NVIDIA RTX 3060+, expect **30–60 FPS**.

### Before vs After Day 8

| Metric | Before | After | Speedup |
|--------|--------|-------|---------|
| Lane detection | 182 ms | 18.6 ms | **9.8×** |
| Total pipeline | 367 ms | 114 ms | **3.2×** |
| FPS | 2.7 | 8.8 | **3.2×** |

**What drives the speedups:**

| Change | Impact |
|--------|--------|
| `FrameResizeStage` (4K → 1280×720) | Lane: 9.8×, Alerts: 6.6× |
| `hough_min_line_length` 25 → 40 px | Removes noise, fixes left/right swap |
| ROI trapezoid narrowed | Excludes road shoulders from Hough |
| `detection.imgsz: 640` explicit | 1.2× detection speedup |
| `depth.skip_frames: 3` | MiDaS only on every 3rd frame |

---

## Known Limitations

### Lane Detection
Classical CV (Canny + Hough) works well on clear, straight roads with visible
lane markings. Known failure modes:

| Scenario | Why it fails | Potential fix |
|----------|-------------|---------------|
| **Sharp curves** | `HoughLinesP` finds only straight segments; fitted line drifts off the marking | Replace with polynomial fitting or an ML-based model (UFLD) |
| **Faded markings** | Weak intensity gradients don't survive `hough_threshold` | Lower `canny_low_threshold` or adaptive thresholding |
| **Night / low-light** | Low contrast → weak Canny edges | CLAHE pre-processing |
| **Heavy shadows** | Shadow edges look like lane lines to Hough | HSV-based shadow removal |
| **Steep hills** | Fixed ROI trapezoid assumes flat road | Dynamic ROI estimation |
| **Wet roads / glare** | Specular reflections flood Hough with false positives | Not addressable in software alone |

### Depth Estimation
- MiDaS produces **relative depth**, not metric (physically-calibrated) depth.
- `calibration_scale: 30.0` in `config.yaml` is a heuristic approximation
  suitable for visual ordering and TTC estimation, not for precise distance reporting.
- Accuracy varies significantly with camera FOV and mounting height.
- To get true metric depth: stereo camera setup or LiDAR fusion.

### TTC / Collision Estimation
- TTC computed from monocular depth is noisy. The linear least-squares smoother
  (10-frame window) substantially reduces noise but does not eliminate it.
- Objects outside the detected ego lane have risk downgraded by one level —
  this is a simple horizontal-centre heuristic, not true multi-lane geometry.
- Minimum 3 history observations required before TTC is reported; newly appeared
  objects show `risk_level: SAFE` until sufficient history exists.

### General
- Single-client WebSocket server — each connection replays the video from the
  start; no shared-state multi-client support.
- No GPU optimizations applied; CUDA support is auto-detected but not tuned.

---

## Configuration Reference

All runtime settings live in `configs/config.yaml`. No hardcoded values in `src/`.

| Key | Type | Description |
|-----|------|-------------|
| `source.type` | `file/webcam` | Input source selector |
| `source.file_path` | string | Path to video file |
| `pipeline.resize_width/height` | int | Pipeline working resolution (1280×720) |
| `detection.model_path` | string | YOLO weights (auto-downloaded) |
| `detection.confidence_threshold` | float | Min detection confidence (0–1) |
| `detection.imgsz` | int | YOLO internal inference size (640) |
| `lanes.canny_low_threshold` | int | Canny edge low threshold |
| `lanes.hough_min_line_length` | int | Min Hough segment length (px) |
| `lanes.smoothing_frames` | int | Frames to average lane lines over |
| `depth.model_name` | string | `midas_small` / `midas_hybrid` / `midas_large` |
| `depth.skip_frames` | int | Run MiDaS every N frames |
| `depth.calibration_scale` | float | Heuristic scale (relative → pseudo-metric) |
| `fusion.history_length` | int | Distance history buffer per track |
| `fusion.ttc_danger_threshold` | float | TTC < this = DANGER (s) |
| `fusion.ttc_caution_threshold` | float | TTC < this = CAUTION (s) |
| `alerts.danger_persist_frames` | int | Consecutive DANGER frames to trigger alert |
| `alerts.clear_persist_frames` | int | Consecutive clear frames to dismiss |
| `alerts.sound_enabled` | bool | Audio beep on alert |
| `visualization.show_fps` | bool | FPS counter in HUD |
| `visualization.show_depth_panel` | bool | Depth heatmap PIP overlay |
| `visualization.show_lane_overlay` | bool | Lane fill + line overlay |

### Demo Mode vs Debug Mode

```yaml
# DEBUG MODE — full information overlay
visualization:
  show_fps: true
  show_depth_panel: true
  show_lane_overlay: true

# DEMO MODE — clean portfolio recording
visualization:
  show_fps: true
  show_depth_panel: false
  show_lane_overlay: false
```

---

## WebSocket API

**Endpoint:** `ws://localhost:8000/ws/stream`

One JSON message per frame:

```json
{
  "frame_b64": "<base64 JPEG string>",
  "metadata": {
    "frame_number": 142,
    "timestamp": 14.328,
    "fps_current": 8.7,
    "lane_offset": -0.12,
    "active_alert": {
      "active": true,
      "message": "⚠ COLLISION RISK — Car #3 | TTC 1.8s",
      "severity": "DANGER",
      "track_id": 3,
      "duration_seconds": 0.6
    },
    "tracked_objects": [
      {
        "track_id": 3,
        "class_name": "car",
        "bbox": [412, 280, 690, 510],
        "confidence": 0.87,
        "estimated_distance_m": 9.2,
        "closing_speed_mps": 5.1,
        "ttc_seconds": 1.8,
        "risk_level": "DANGER",
        "in_ego_lane": true
      }
    ]
  }
}
```

Full schema: `src/api/schemas.py` or `http://localhost:8000/docs` (Swagger UI).

---

## 10-Day Build Log

| Day | Feature |
|-----|---------|
| ✅ 1 | Project scaffold + modular video pipeline |
| ✅ 2 | Object detection (YOLOv8n) + multi-object tracking (ByteTrack) |
| ✅ 3 | Lane detection (Canny + Hough Transform) |
| ✅ 4 | Monocular depth estimation (MiDaS) + per-object pseudo-metric distance |
| ✅ 5 | Temporal fusion — closing speed, TTC & collision risk |
| ✅ 6 | Alert system — debounced warnings, visualization polish, demo mode |
| ✅ 7 | FastAPI backend — WebSocket stream (annotated frames + JSON metadata) |
| ✅ 8 | FPS optimization (3.2×) + lane detection fix via pipeline resize stage |
| ✅ 9 | React frontend — live dashboard consuming WebSocket stream |
| ✅ 10 | Final integration, documentation overhaul, demo recording |

---

## Recording a Demo

1. Set `visualization.show_depth_panel: false` and `show_lane_overlay: false`
   in `config.yaml` for a clean look.
2. Start the backend: `python src/run_server.py`
3. Open the dashboard at `http://localhost:5173`
4. Use OBS, ShareX, or Windows' built-in `Win + G` Game Bar to record a
   30–60 second clip showing:
   - Detection boxes updating in real time
   - Side panel object list with risk levels
   - A natural DANGER alert triggering (approach a vehicle closely in the clip)
5. Save as `.mp4` and link in your portfolio.
