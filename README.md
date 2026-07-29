# AI Driving Assistant 🚗

A real-time computer vision pipeline for autonomous driving assistance.
Built incrementally over 10 days.

| Day | Status | Feature |
|-----|--------|---------|
| ✅ 1 | Done | Project scaffold + video pipeline |
| ✅ 2 | Done | Object detection (YOLOv8n) + Multi-object tracking (ByteTrack) |
| ✅ 3 | Done | Lane detection (Canny + Hough Transform) |
| ✅ 4 | Done | Monocular depth estimation (MiDaS) + per-object distance |
| ✅ 5 | Done | Temporal fusion — closing speed, TTC & collision risk |
| ✅ 6 | Done | Alert system — debounced warnings, visualization polish, demo mode |
| ✅ 7 | Done | FastAPI backend — WebSocket stream (annotated frames + JSON metadata) |
| ✅ 8 | Done | FPS optimization + lane detection fix (pipeline resize stage) |
| 9   |      | Frontend dashboard (React/vanilla JS WebSocket consumer) |
| 10  |      | Final integration & demo |

---

## Project Structure

```
ai-driving-assistant/
├── src/
│   ├── main.py                    # Entry point — wires the whole pipeline
│   ├── run_server.py              # FastAPI server launcher (Day 7)
│   ├── api/                       # Day 7 ──────────────────────────────
│   │   ├── app.py                 # FastAPI app instance + CORS + /health
│   │   ├── websocket_handler.py   # /ws/stream — full pipeline over WebSocket
│   │   └── schemas.py             # Pydantic models (FramePayload, etc.)
│   ├── pipeline/
│   │   ├── video_source.py        # Webcam / video file abstraction
│   │   ├── frame_processor.py     # Pluggable stage-list processing engine
│   │   └── resize_stage.py        # Day 8: FrameResizeStage (pre-pipeline resize)
│   ├── detection/                 # Day 2 ──────────────────────────────
│   │   ├── detector.py            # YOLOv8 wrapper (VehicleDetector)
│   │   ├── tracker.py             # ByteTrack wrapper (VehicleTracker)
│   │   └── stage.py               # DetectionStage: the pipeline adapter
│   ├── lanes/                     # Day 3 ──────────────────────────────
│   │   ├── lane_detector.py       # Classical CV lane detection
│   │   ├── lane_utils.py          # Lane math helpers
│   │   └── stage.py               # LaneDetectionStage: pipeline adapter
│   ├── depth/                     # Day 4 ──────────────────────────────
│   │   ├── depth_estimator.py     # MiDaS monocular depth wrapper
│   │   ├── depth_utils.py         # Colormap, distance extraction, calibration
│   │   └── stage.py               # DepthEstimationStage: pipeline adapter
│   ├── fusion/                    # Day 5 ──────────────────────────────
│   │   ├── object_history.py      # Per-track distance/time rolling buffer
│   │   ├── collision_estimator.py # TTC computation + risk classification
│   │   └── stage.py               # CollisionFusionStage: pipeline adapter
│   ├── alerts/                    # Day 6 ──────────────────────────────
│   │   ├── alert_manager.py       # Priority, debounce & hysteresis
│   │   ├── sound_alert.py         # Programmatic beep (numpy sine wave)
│   │   └── stage.py               # AlertStage: banner + label polish
│   ├── utils/
│   │   ├── config.py              # YAML config loader + validator
│   │   └── logger.py              # Centralized logging setup
│   └── visualization/
│       └── display.py             # HUD, FPS overlay, alert-aware bounding boxes
├── configs/
│   └── config.yaml                # All runtime settings live here
├── models/                        # Downloaded .pt weights go here (gitignored)
├── data/
│   └── sample_videos/             # Drop .mp4 / .avi test clips here
└── requirements.txt
```

---

## Quick Start

### 1. Install PyTorch (CPU-only — no GPU required)

> **⚠️ Do this FIRST** to get the lightweight CPU build (~124 MB) instead of
> the CUDA build (~2.5 GB):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 2. Install all other dependencies

```bash
pip install -r requirements.txt
```

### 3. Add a sample video

Drop any `.mp4` driving clip into `data/sample_videos/` and set in
`configs/config.yaml`:
```yaml
source:
  type: "file"
  file_path: "data/sample_videos/your_clip.mp4"
```

### 4. Run

```bash
python src/main.py
```

On first run, `yolov8n.pt` (~6 MB) and MiDaS small (~2 MB) weights are
downloaded automatically. Press **`q`** or close the window to exit.

### Switch to Webcam

In `configs/config.yaml`:
```yaml
source:
  type: "webcam"
  webcam_index: 0
```
No code changes needed.

---

## Expected FPS (Benchmarks)

All measurements taken on **CPU-only** hardware (no GPU). Results will be
significantly better on any NVIDIA GPU via CUDA.

### Before vs After Day 8 optimizations

**Hardware:** Intel CPU, no GPU — 3840×2160 source video → 1280×720 pipeline working resolution

| Stage | Before (ms) | After (ms) | Speedup | Notes |
|-------|------------|-----------|---------|-------|
| *(resize)* | — | 6.3 | — | New stage 0, trivial cost |
| detection | 68.5 | 55.8 | 1.2× | Smaller input + explicit `imgsz=640` |
| **lanes** | **182.0** | **18.6** | **9.8×** | Root fix: Canny+Hough on 720p not 4K |
| depth | 87.6 | 28.5 | 3.1× | skip_frames=3 cache; smaller input to transform |
| fusion | 0.3 | 0.3 | — | Pure math, always negligible |
| alerts | 28.4 | 4.3 | 6.6× | Smaller frame = faster draw calls |
| **TOTAL** | **366.9** | **113.7** | **3.2×** | |
| **FPS** | **2.73** | **8.79** | **3.2×** | |

> **"Good" CPU FPS context:** At 8-9 FPS CPU-only with YOLO + MiDaS + all overlays,
> this is at or above the practical ceiling for CPU inference on a full CV pipeline.
> On an NVIDIA RTX 3060+, expect 30-60 FPS.

### What changes drive each speedup

| Change | File | Impact |
|--------|------|--------|
| `FrameResizeStage` (4K → 1280×720) | `resize_stage.py` + `main.py` | **Lane: 9.8×, Alerts: 6.6×** |
| `hough_min_line_length` 25 → 40 px | `config.yaml` | Removes noise hits, fixes left/right swap |
| ROI trapezoid narrowed at bottom | `config.yaml` | Excludes road shoulders from Hough input |
| `detection.imgsz: 640` explicit | `detector.py` + `config.yaml` | 1.2× detection speedup |
| `depth.skip_frames: 3` (existing) | `config.yaml` | MiDaS only runs every 3rd frame |

---

## Lane Detection — Known Limitations

The lane detector uses **classical computer vision** (Canny + Hough) and works
best on clear, straight roads with visible painted lane markings. Known failure modes:

| Scenario | Why it fails | Mitigation |
|----------|-------------|------------|
| **Sharp curves** | `HoughLinesP` finds straight segments only. On a curve, the fitted line is a chord through the arc and drifts off the marking. | Replace with polynomial fitting or an ML-based lane model (e.g., UFLD). |
| **Faded / worn markings** | Canny relies on intensity gradients. Faint markings produce weak edges that don't survive the `hough_threshold`. | Lower `canny_low_threshold` or switch to adaptive thresholding. |
| **Night / low-light footage** | Low contrast → weak Canny edges throughout the ROI. | Requires histogram equalization or CLAHE pre-processing. |
| **Heavy shadows** | Strong shadow edges cross the ROI and look like lane lines to Hough. | Adaptive thresholding or HSV-based shadow removal. |
| **Steep hills / uphill slopes** | The fixed ROI trapezoid assumes a flat road. Horizon rises on hills, clipping actual lane lines. | Dynamic ROI estimation (out of scope for classical CV). |
| **Wet roads / glare** | Specular reflections create high-contrast edges everywhere, flooding Hough with false positives. | Polarization filtering; not addressable in software alone. |
| **Lane changes** | Only the innermost visible lane per side is selected. Adjacent lanes bleed in during a lane change. | Requires multi-lane geometry modeling. |

---

## How the FrameProcessor Stage-List Works

`FrameProcessor` maintains an ordered list of **stages**. Each stage is a
callable with the signature:

```python
(frame: np.ndarray, meta: dict) -> (frame: np.ndarray, meta: dict)
```

- `frame` is the current BGR image.
- `meta` is a shared dictionary that accumulates results across stages.

Stages are chained: the output of stage N is the input to stage N+1.

### Meta dict keys (growing each day)

| Stage added   | Key written to meta      | Read by future stages       |
|---------------|--------------------------|-----------------------------:|
| Day 2 detect  | `tracked_objects`        | depth, fusion, alerts       |
| Day 3 lanes   | `lane_lines`, `lane_offset` | fusion, alerts          |
| Day 4 depth   | `depth_map`              | fusion, alerts              |
| Day 5 fusion  | enriches each TrackedObject with `closing_speed_mps`, `ttc_seconds`, `risk_level`, `in_ego_lane` | alerts |

### ⚠️ Stage ordering matters

```python
# main.py — stage registration order (Day 8)
processor.add_stage("resize",    FrameResizeStage(1280, 720))              # 0th — pre-scale
processor.add_stage("detection", DetectionStage(config["detection"]))      # 1st
processor.add_stage("lanes",     LaneDetectionStage(config["lanes"]))      # 2nd
processor.add_stage("depth",     DepthEstimationStage(config["depth"]))    # 3rd — needs tracked_objects
processor.add_stage("fusion",    CollisionFusionStage(config["fusion"]))   # 4th — needs ALL above
```

`CollisionFusionStage` is the first stage that depends on **all** prior stages:
it reads tracked objects (Day 2), lane lines (Day 3), and per-object distance
(Day 4). Moving it before any of these produces missing data and broken risk
estimates.

---

## Day 5: Temporal Fusion — Closing Speed, TTC & Collision Risk

### How it works (end-to-end)

The fusion pipeline turns per-frame depth snapshots into temporal trends:

```
Frame N:  Tracked Object #7 at ~20 m
Frame N+1: Object #7 at ~19.5 m
Frame N+2: Object #7 at ~19.1 m
   ...           ...           ...
Frame N+9: Object #7 at ~16.8 m

→ History buffer stores last 10 (timestamp, distance) pairs
→ Linear least-squares fit: slope = -3.2 m/s → closing speed = 3.2 m/s
→ TTC = 16.8 m / 3.2 m/s = 5.25 s → CAUTION (< 4s threshold not met → SAFE, but close)
→ Object center is between lane lines → in_ego_lane = True → retain full risk level
```

### 1. History buffer (`object_history.py`)

`ObjectHistoryTracker` maintains a `collections.deque(maxlen=N)` per ByteTrack
ID, storing `(timestamp, distance_m)` pairs. Key design choices:

- **Bounded memory**: `deque(maxlen=10)` evicts the oldest entry automatically.
  Total memory is O(active_tracks × N), constant regardless of video length.
- **Timeout expiration**: Tracks not updated for `history_timeout_seconds` are
  purged. This prevents stale IDs from lingering after an object leaves the frame.
- **Per-track isolation**: Each ByteTrack ID has its own independent history.
  A new ID starts fresh with zero history; it needs `min_history_points`
  observations before TTC estimation begins.

### 2. Closing speed via linear fit (`collision_estimator.py`)

**Why a linear fit across N frames, not just `Δd/Δt` from the last two?**

Monocular depth estimates (MiDaS) are inherently noisy. Two consecutive frames
might yield distances of 18.3 m and 17.9 m — a 0.4 m drop in ~33 ms, implying
a closing speed of 12 m/s (43 km/h). The very next pair might read 17.9 m and
18.5 m, suggesting the object is *retreating* at 18 m/s. Both are noise.

A **least-squares linear fit** across the full history window:

```
distance = slope × time + intercept
slope = (n·Σ(tᵢdᵢ) − Σtᵢ·Σdᵢ) / (n·Σ(tᵢ²) − (Σtᵢ)²)
```

- Minimizes sum of squared residuals → each noisy sample has limited influence.
- The slope represents the *average trend*, far more stable than any single Δd/Δt.
- Computable in O(N) via the closed-form normal equation — negligible cost.
- No tuning knobs beyond window size (unlike exponential moving averages).

This is the same smoothing principle behind Kalman filters, but simpler and
fully transparent.

### 3. TTC formula and guards

```
TTC = current_distance / closing_speed
```

Where:
- `current_distance` = most recent depth observation (meters)
- `closing_speed` = −slope from the linear fit (positive when approaching)

**Guards against non-positive closing speed:**

| Scenario | closing_speed | TTC | Risk |
|----------|--------------|-----|------|
| Object approaching | +3.2 m/s | 5.25 s | Based on thresholds |
| Object stationary | ≈ 0 m/s | ∞ (None) | SAFE |
| Object retreating | −2.1 m/s | Negative (None) | SAFE |

Dividing by zero (stationary) would crash. Negative TTC (retreating) is
physically meaningless — you can't collide with something moving away.
We threshold at `closing_speed > 0.1 m/s` to also filter near-zero noise.

### 4. Risk thresholds

| TTC value | Risk level | Default threshold |
|-----------|-----------|-------------------|
| < 2.0 s | **DANGER** | `ttc_danger_threshold` |
| 2.0 – 4.0 s | **CAUTION** | `ttc_caution_threshold` |
| > 4.0 s or None | **SAFE** | — |

All thresholds are configurable in `config.yaml`. No magic numbers in code.

### 5. Lane-relevance filtering

A car three lanes over closing at 5 m/s is very different from one directly
ahead. We check whether each object's horizontal center falls between the
detected ego lane lines (interpolated at the object's y-coordinate).

- **In ego lane**: Retain full risk level.
- **Not in ego lane**: Downgrade by one level (DANGER → CAUTION, CAUTION → SAFE).

This significantly reduces false alarm fatigue. It's NOT precise multi-lane
geometry — just a practical "is this in my path?" filter.

### 6. Visualization

Each tracked object's bounding box is color-coded by risk level:
- 🟢 **Green** = SAFE
- 🟡 **Yellow** = CAUTION  
- 🔴 **Red** = DANGER

Labels show: `"Car #7 ~18m, closing 3.2 m/s, TTC 5.6s [SAFE]"`

DANGER + in_ego_lane objects get **extra-thick borders** with a red inner
outline — this is the single most critical visual signal in the system.

### FPS impact

The fusion stage performs only:
- One `deque.append()` per tracked object per frame
- One O(N) linear regression per tracked object (N ≤ 10)
- A few comparisons for risk classification

This is pure math on tiny buffers — **negligible compute cost** compared to
YOLO inference or MiDaS depth estimation. FPS impact is < 1%.

---

## Day 6: Alert System — Debounce, Priority & Visualization Polish

### Why debounce/hysteresis is essential for driver-facing alerts

A raw threshold check fires the alert banner the moment TTC < 2.0 s and clears
it the moment TTC ≥ 2.0 s. With noisy monocular depth, TTC might oscillate
between 1.9 s and 2.1 s on consecutive frames even for a steadily approaching
object. Without debouncing, the banner flickers on/off at 15 fps.

**Flickering warnings are worse than no warning at all:**

1. **Alarm fatigue**: After 20 flickers in 5 seconds, the driver ignores the
   warning entirely. When a real threat triggers a sustained alert, they no
   longer respond.
2. **Trust degradation**: A system that cries wolf gets disabled by operators.
   This is the #1 cause of ADAS safety feature bypasses in production vehicles
   (Mobileye targets < 1 false alarm per 10 000 km for this reason).
3. **Cognitive load**: Visual flicker increases driver reaction time to real
   events, the exact opposite of the system's purpose.

**The solution: two-counter hysteresis**

```
State machine:
  [IDLE]  ─ DANGER holds for danger_persist_frames (default 5) ─►  [ACTIVE]
  [ACTIVE] ─ condition clears for clear_persist_frames (default 10) ─► [IDLE]
```

- **5 frames to trigger** (at 30 fps = 167 ms): Filters single-frame noise.
- **10 frames to clear** (at 30 fps = 333 ms): Keeps alert visible while the
  driver registers and begins reacting. Asymmetry is intentional — safer to
  display slightly longer than to vanish while the driver is still processing.

This exact pattern is used in industrial PLC safety interlocks and automotive
ABS wheel-slip detection. It's a well-understood, auditable safety mechanism.

### Demo mode vs debug mode

Toggle these three flags in `configs/config.yaml` without any code changes:

```yaml
# DEBUG MODE (development / tuning)
visualization:
  show_fps: true           # FPS counter in top-left HUD
  show_depth_panel: true   # Depth heatmap PIP in bottom-right
  show_lane_overlay: true  # Translucent lane fill + lane lines

# DEMO MODE (clean portfolio recording)
visualization:
  show_fps: true           # Keep FPS to show performance
  show_depth_panel: false  # Hide depth clutter
  show_lane_overlay: false # Hide lane fill for cleaner look
```

The flags flow through the pipeline without touching any stage code:
`config.yaml` → `main.py` (injects into stage configs) → stages gate their
draw calls → `AlertStage` writes `meta["viz_config"]` → `draw_hud()` reads it.

### Complete pipeline (as of Day 6)

```
VideoSource (file/webcam)
    ↓
DetectionStage       — YOLO inference + ByteTrack IDs → meta["tracked_objects"]
    ↓
LaneDetectionStage   — Canny + Hough → meta["lane_lines"], meta["lane_offset"]
    ↓
DepthEstimationStage — MiDaS → obj.estimated_distance_m  (per-object)
    ↓
CollisionFusionStage — linear-fit TTC → obj.risk_level, obj.ttc_seconds, obj.in_ego_lane
    ↓
AlertStage           — debounced alert → meta["active_alert"] + banner + polished labels
    ↓
show_frame()         — HUD + display (reads meta["viz_config"] for toggles)
```

All six stages are fully operational in a single `python src/main.py` run.

---

## Configuration Reference (`configs/config.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `source.type` | `file/webcam` | Input source selector |
| `source.file_path` | string | Path to video file |
| `source.webcam_index` | int | OS device index for webcam |
| `pipeline.resize_width` | int/null | Pipeline working width (default 1280) |
| `pipeline.resize_height` | int/null | Pipeline working height (default 720) |
| `display.width/height` | int/null | Output window size |
| `display.window_title` | string | OpenCV window title |
| `detection.model_path` | string | YOLO weights file (auto-downloaded) |
| `detection.confidence_threshold` | float | Min detection confidence (0–1) |
| `detection.classes` | list | COCO classes to detect |
| `detection.device` | string | `auto`, `cpu`, or `cuda` |
| `detection.imgsz` | int | YOLO internal inference resolution (default 640) |
| `detection.tracker.lost_track_buffer` | int | Frames before a track is deleted |
| `detection.tracker.frame_rate` | int | Source FPS for Kalman motion model |
| `lanes.canny_low_threshold` | int | Canny edge low threshold |
| `lanes.canny_high_threshold` | int | Canny edge high threshold |
| `lanes.hough_min_line_length` | int | Min Hough segment length in pixels (at pipeline resolution) |
| `lanes.smoothing_frames` | int | Frames to average lane lines over |
| `depth.model_name` | string | `midas_small`, `midas_hybrid`, or `midas_large` |
| `depth.input_resolution` | int | Internal inference size (256 recommended for CPU) |
| `depth.device` | string | `auto`, `cpu`, or `cuda` |
| `depth.calibration_scale` | float | Heuristic scale for pseudo-metric conversion |
| `depth.skip_frames` | int | Run MiDaS every N frames, cache otherwise |
| `fusion.history_length` | int | Frames of distance history per track (default 10) |
| `fusion.history_timeout_seconds` | float | Expire stale histories after N seconds |
| `fusion.ttc_danger_threshold` | float | TTC below this = DANGER (seconds) |
| `fusion.ttc_caution_threshold` | float | TTC below this = CAUTION (seconds) |
| `fusion.min_history_points` | int | Min observations before computing TTC |
| `alerts.danger_persist_frames` | int | Consecutive DANGER frames to trigger alert |
| `alerts.clear_persist_frames` | int | Consecutive clear frames to dismiss alert |
| `alerts.sound_enabled` | bool | Play beep on new alert (`pip install sounddevice`) |
| `visualization.show_fps` | bool | Toggle FPS counter in HUD |
| `visualization.show_depth_panel` | bool | Toggle depth heatmap PIP overlay |
| `visualization.show_lane_overlay` | bool | Toggle lane fill + lane lines |
| `logging.level` | string | DEBUG/INFO/WARNING/ERROR |
| `logging.log_file` | string/null | Log file path (null = console only) |

