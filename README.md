# AI Driving Assistant 🚗

A real-time computer vision pipeline for autonomous driving assistance,
built from scratch in 10 days (plus a validation & hardening pass). The system
detects road objects, tracks them across frames, finds the ego lane, estimates
each object's distance, computes time-to-collision, raises debounced collision
alerts, and streams everything — annotated frames + structured JSON metadata —
over WebSocket to a React dashboard.

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
│  Canny + ROI polygon + HoughLinesP → vanishing-point filter →   │
│  innermost marking per side → ego-lane corridor (detected or    │
│  default) + lane offset                                         │
│  meta["lane_lines"], meta["ego_lane"], meta["vanishing_point"]  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ + lane_lines, lane_offset
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3 — DepthEstimationStage                                  │
│  Ground-plane ranging: Z = f·H / (box bottom − horizon row)     │
│  → obj.estimated_distance_m  (MiDaS small: heatmap + fallback)  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ + depth per object
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4 — CollisionFusionStage                                  │
│  Rolling distance buffer (10 frames, video timestamps) →        │
│  least-squares closing speed → TTC → SAFE/CAUTION/DANGER       │
│  + in_ego_lane (box ground-contact point vs. ego corridor)      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ + risk_level, ttc_seconds per object
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 5 — AlertStage                                            │
│  Debounced hysteresis (5 frames trigger, 10 frames clear) →     │
│  meta["active_alert"] + banner + the single risk-aware label    │
│  layer on the frame                                             │
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
| Lane Detection | Classical CV — Canny + HoughLinesP + vanishing-point filtering (OpenCV) |
| Distance Estimation | Ground-plane monocular ranging (camera height + FOV + horizon) |
| Depth Heatmap | [MiDaS small](https://github.com/isl-org/MiDaS) via `torch.hub` (also a selectable distance fallback) |
| Temporal Fusion | Custom linear least-squares TTC estimator |
| Alert System | Two-counter debounce/hysteresis state machine |
| Backend API | [FastAPI](https://fastapi.tiangolo.com/) + WebSocket stream |
| Frontend | [React 18](https://react.dev/) + [Vite 5](https://vitejs.dev/) |
| Tests | [pytest](https://pytest.org/) — geometry, TTC, ego-lane and alert logic |
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
│   │   ├── depth_utils.py         # Ground-plane ranging, colormap, MiDaS distance
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
├── tests/                         # pytest suite (no models/video needed)
├── configs/
│   └── config.yaml                # All runtime settings + camera presets
├── models/                        # Downloaded .pt weights (gitignored)
├── data/
│   └── sample_videos/             # Drop .mp4 clips here
├── requirements.txt
└── requirements-dev.txt           # + pytest
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

### 3. Calibrate for your camera (once per camera mounting)

Lane detection and distance estimation depend on where the camera sits. Four
values in `config.yaml` describe it; presets for both sample clips are in the
comments there (the default config is set up for `sample2.mp4`):

| Key | What it is | How to find it |
|-----|-----------|----------------|
| `lanes.vanishing_point` | Where the lane markings meet on the horizon, as `[x, y]` fractions of the frame | Set `lanes.debug_overlay: true` — a magenta cross marks it; move it onto the point where the lane lines converge |
| `lanes.roi_polygon` | Road area searched for lane markings | Must stay below the horizon and above the car's hood (cyan outline in debug mode) |
| `depth.camera_height_m` | Camera height above the road | `lane_width_m × (row − horizon_row) / lane_width_px` at any row (US interstate lane = 3.66 m) |
| `depth.camera_hfov_deg` | Horizontal field of view | From the camera spec, or from dashed-lane spacing (dash + gap = 12.2 m on US interstates) |

### 4. Run: OpenCV window mode (no frontend needed)

```bash
python src/main.py
```

Press **`q`** to quit. YOLO and MiDaS weights auto-download on first run.

### 5. Run: WebSocket server mode + React dashboard

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
panels appear automatically. The dashboard connects to `/ws/stream` on its own
origin and Vite proxies it to the backend, so only the dashboard's port needs
to be reachable — e.g. `npm run dev -- --host` to open it from a phone on your
network. To point it at a backend elsewhere, set `VITE_WS_URL`
(e.g. `VITE_WS_URL=ws://192.168.1.20:8000/ws/stream npm run dev`).

For a production build: `npm run build && npm run preview` (port 4173, same proxy).

### 6. Webcam mode

```yaml
source:
  type: "webcam"
  webcam_index: 0
```

No code changes needed — but calibrate the camera values above for it.

### 7. Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

The suite covers lane geometry (including the left/right crossing bug),
ground-plane ranging, closing-speed/TTC estimation, ego-lane membership and
the alert hysteresis. It runs in about a second and needs no models or video.

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
>
> CPU timings are very sensitive to machine load: the same build measured
> ~120 ms/frame on an idle machine and 400–850 ms/frame while other heavy
> processes ran. Because fusion uses video time, slow processing no longer
> distorts closing speed or TTC — it only lowers the displayed frame rate.
> With `distance_method: "ground_plane"` and `show_depth_panel: false`,
> MiDaS doesn't run at all, removing the depth stage's cost.

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
| **Steep hills** | Fixed vanishing point / ROI assume a flat road | Online vanishing-point estimation |
| **Wet roads / glare** | Specular reflections flood Hough with false positives | Not addressable in software alone |

When markings aren't visible, the ego-lane corridor falls back to default
lines through the vanishing point (`lanes.default_lane_bottom_x`), so
collision risk stays lane-aware.

### Distance Estimation
- **Ground-plane ranging** (default) assumes a flat road and a calibrated
  camera height / FOV. A ±1 px error in the box bottom is a relative error of
  `1 / (pixels below horizon)`: precise up close, coarse near the horizon.
  Hills and pitching (braking) shift the true horizon and bias it.
- Boxes clipped by the bottom of the frame read slightly too far.
- **MiDaS** (`distance_method: "midas"`) produces *affine-invariant* relative
  inverse depth — unknown scale **and** shift. Measured on `sample2.mp4`, its
  per-object values barely track real distance (log-log slope ≈ −0.15 vs the
  ideal −1), so it is only used for the heatmap and as a fallback.
- True metric depth needs stereo, LiDAR/radar, or a metric depth model.

### TTC / Collision Estimation
- TTC computed from monocular depth is noisy. The linear least-squares smoother
  (10-frame window) substantially reduces noise but does not eliminate it.
- Objects outside the ego lane have risk downgraded by one level — the test
  uses the box's bottom-centre (ground contact point) against straight lane
  lines, not true multi-lane geometry.
- Closing speed is measured in **video time** for files (frame index / fps),
  so TTC is correct even though CPU processing is slower than real time.
- Minimum 3 history observations required before TTC is reported; newly appeared
  objects show `risk_level: SAFE` until sufficient history exists.

### General
- Single-client WebSocket server — each connection replays the video from the
  start; no shared-state multi-client support.
- Vehicles appear one frame after first detection: ByteTrack must confirm a
  track before it gets an ID (unconfirmed detections have no stable identity
  for distance history).
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
| `lanes.vanishing_point` | [x, y] | Road vanishing point (fractions of frame) — camera-specific |
| `lanes.vp_tolerance` | float | Max miss distance of a lane segment from the VP (fraction of width) |
| `lanes.roi_polygon` | list | Road region searched for markings (any number of points) |
| `lanes.default_lane_bottom_x` | [l, r] | Fallback ego corridor when markings aren't detected |
| `lanes.min_abs_slope` | float | Min \|dy/dx\| of a lane segment (wide-angle cameras need ~0.2) |
| `lanes.canny_low_threshold` | int | Canny edge low threshold |
| `lanes.hough_min_line_length` | int | Min Hough segment length (px) |
| `lanes.smoothing_frames` | int | Frames to average lane lines over |
| `lanes.max_missed_frames` | int | Drop a lane line after this many frames undetected |
| `depth.distance_method` | string | `ground_plane` (default) or `midas` |
| `depth.camera_height_m` | float | Camera height above the road (ground_plane) |
| `depth.camera_hfov_deg` | float | Horizontal field of view (ground_plane) |
| `depth.model_name` | string | `midas_small` / `midas_hybrid` / `midas_large` |
| `depth.skip_frames` | int | Run MiDaS every N frames |
| `depth.calibration_scale` | float | Heuristic scale for the MiDaS method (relative → pseudo-metric) |
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

- `timestamp` is the frame's capture time in seconds — its position in the
  video for file sources.
- `lane_offset` is `null` when both lane lines aren't detected.
- `estimated_distance_m` is `null` for objects at or above the horizon
  (beyond reliable ground-plane range).
- `closing_speed_mps` / `ttc_seconds` are `null` until enough distance history
  exists, when the object isn't approaching, or when the measured speed is
  physically implausible (noise).
- The server closes the socket with code `1000` when the video ends; the
  dashboard then shows **Stream ended** instead of reconnecting.

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
| ✅ 11 | Validation on real footage & hardening — see below |

### Day 11: what validating on the sample footage uncovered

Running the full pipeline headless over `sample2.mp4` and checking the
numbers showed the collision-warning path never actually fired. Root causes
and fixes:

| Problem | Root cause | Fix |
|---------|-----------|-----|
| Lanes found in 5% of frames, left/right lines crossing in an X | Wide-angle lane lines fell outside the narrow ROI trapezoid; road cracks/shadows with the "right" slope sign dominated the fit | Vanishing-point consistency filter, left/right half-plane split, innermost-marking selection, length-weighted fit, wider ROI polygon, per-camera presets |
| `in_ego_lane` almost never true → alerts impossible | Required both lane lines | Default ego corridor through the vanishing point when markings aren't visible; ground-contact point (box bottom) for the test |
| TTC ~10–20× too long | Closing speed used wall-clock time, but CPU processing runs far slower than the video | Fusion uses video time (frame index / fps) for file sources |
| Cars at ~200 m | MiDaS relative depth with a guessed scale; uncorrelated with distance | Ground-plane ranging with a measured camera height / FOV |
| False alarm on a distant truck (phantom 30–70 m/s closing speed) | Box bottom at the horizon (road climbing ahead) → fell back to noisy MiDaS distance | No distance beyond the horizon; closing speeds above 40 m/s treated as noise |
| False alarm on a semi truck alongside (`sample.mp4`) | Its box reached down onto its reflection in the hood: fake ground contact, jittering distance, and an over-wide extrapolated corridor | Contact points clamped to the lowest visible road row (`meta["road_bottom_y"]`) |
| Analysis on painted overlays | Stages drew boxes and the lane fill in-place before later stages analysed the frame | Analysis stages read an un-annotated `meta["clean_frame"]` |
| Three stacked labels per object, "#-1" IDs | Detection, fusion and alert stages each drew labels; unconfirmed tracks reported | One label layer (AlertStage); unconfirmed tracks skipped |
| Black box in demo mode | Alert stage "erased" a depth PIP that was never drawn | Removed |

**Measured on the sample clips (full pipeline, headless):**

| Metric | `sample2.mp4` before | `sample2.mp4` after | `sample.mp4` after (preset) |
|--------|---------------------|---------------------|-----------------------------|
| Frames with both lane lines | 5% | 98% | 92% |
| Median object distance | 183 m (most clamped at 200 m) | 59 m | 43 m |
| Median TTC (approaching objects) | 59.5 s | 5.2 s | 14.3 s |
| Risk levels seen | 100% SAFE | SAFE / CAUTION | SAFE / CAUTION |
| Collision alerts | 0 (system could not fire) | 0 | 0 |

Neither clip contains a real near-collision — both are routine highway
driving — so zero alerts is the correct result, and it was only reached after
fixing the two false alarms above. The alert path itself (DANGER → debounce →
banner) is covered by unit tests with synthetic approaching tracks; validating
it on footage with a genuine hard-braking event is the obvious next step.

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
