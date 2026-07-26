# AI Driving Assistant 🚗

A real-time computer vision pipeline for autonomous driving assistance.
Built incrementally over 10 days.

| Day | Status | Feature |
|-----|--------|---------|
| ✅ 1 | Done | Project scaffold + video pipeline |
| ✅ 2 | Done | Object detection (YOLOv8n) + Multi-object tracking (ByteTrack) |
| ✅ 3 | Done | Lane detection (Canny + Hough Transform) |
| ✅ 4 | Done | Monocular depth estimation (MiDaS) + per-object distance |
| 5   |      | Multi-module fusion engine |
| 6   |      | Collision risk warnings & alerts |
| 7   |      | Performance optimization |
| 8   |      | Dashboard / HUD overlay |
| 9   |      | Recording + output video |
| 10  |      | Final integration & demo |

---

## Project Structure

```
ai-driving-assistant/
├── src/
│   ├── main.py                    # Entry point — wires the whole pipeline
│   ├── pipeline/
│   │   ├── video_source.py        # Webcam / video file abstraction
│   │   └── frame_processor.py    # Pluggable stage-list processing engine
│   ├── detection/                 # Day 2 ──────────────────────────────
│   │   ├── detector.py            # YOLOv8 wrapper (VehicleDetector)
│   │   ├── tracker.py             # ByteTrack wrapper (VehicleTracker)
│   │   └── stage.py              # DetectionStage: the pipeline adapter
│   ├── lanes/                     # Day 3 ──────────────────────────────
│   │   ├── lane_detector.py       # Classical CV lane detection
│   │   ├── lane_utils.py          # Lane math helpers
│   │   └── stage.py              # LaneDetectionStage: pipeline adapter
│   ├── depth/                     # Day 4 ──────────────────────────────
│   │   ├── depth_estimator.py     # MiDaS monocular depth wrapper
│   │   ├── depth_utils.py         # Colormap, distance extraction, calibration
│   │   └── stage.py              # DepthEstimationStage: pipeline adapter
│   ├── utils/
│   │   ├── config.py              # YAML config loader + validator
│   │   └── logger.py              # Centralized logging setup
│   └── visualization/
│       └── display.py             # FPS overlay + bounding box drawing
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

| Hardware | Day 1 | Day 2 (YOLO) | Day 3 (+Lanes) | Day 4 (+Depth) |
|----------|-------|--------------|-----------------|-----------------|
| **CPU only** (i5/i7) | ~60+ | 8–15 | 8–14 | **4–8** |
| **NVIDIA GPU** (RTX 3060+) | ~200+ | 60–120 | 55–110 | **30–60** |

> Depth estimation is typically the heaviest single-stage compute cost. Using
> `input_resolution: 256` instead of 384+ is the primary lever to trade depth
> quality for speed. See the "FPS Impact" section below.

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
| Day 5 fusion  | `risk_scores`            | alerts                      |

### ⚠️ Stage ordering matters

```python
# main.py — stage registration order
processor.add_stage("detection", DetectionStage(config["detection"]))
processor.add_stage("lanes",     LaneDetectionStage(config["lanes"]))
processor.add_stage("depth",     DepthEstimationStage(config["depth"]))  # MUST be after detection
```

`DepthEstimationStage` reads `meta["tracked_objects"]` to compute per-object
distances. If registered before `DetectionStage`, it would find no objects
and produce no distance estimates.

---

## Day 4: Monocular Depth Estimation — Technical Details

### Why monocular depth is RELATIVE, not metric

A single 2D image lacks the geometric information needed to recover absolute
distances. Specifically:

- **No parallax**: Stereo vision works by comparing two images taken from
  slightly different viewpoints (known baseline distance). A single camera
  provides only one viewpoint — there's no triangulation possible.

- **No time-of-flight**: LiDAR and radar measure how long a signal takes to
  bounce back from an object, giving direct range. A camera captures light
  intensity, not travel time.

- **Scale ambiguity**: A small object close up and a large object far away
  can produce identical images. Without knowing the actual size of objects
  or the camera's physical parameters, the model cannot distinguish between
  these cases.

What monocular depth models (like MiDaS) *do* learn is **ordinal depth** —
"object A is closer than object B" — from training on millions of images
with depth supervision. The output is a dense map where relative values
indicate depth ordering, but the absolute numbers are arbitrary.

**To get real-world metric depth, you need:**

| Method | How it works | Accuracy |
|--------|-------------|----------|
| Stereo cameras | Triangulation from two viewpoints with known baseline | ±2–5 cm at short range |
| LiDAR | Time-of-flight laser pulses | ±2 cm |
| Calibrated mono | Known camera intrinsics + ground-plane assumption | ±10–30%, fragile |

Our `calibration_scale` heuristic in `depth_utils.py` provides a rough
approximation suitable for relative comparisons ("this car is closer than
that truck"), but NOT for safety-critical absolute distance measurements.

### Why median (not mean) depth within a bounding box

Bounding boxes are axis-aligned rectangles that include background pixels
around the actual object silhouette:

```
┌───────────────────┐
│  background       │
│   ┌───────────┐   │
│   │  actual   │   │
│   │  object   │   │
│   └───────────┘   │
│  background       │
└───────────────────┘
     bounding box
```

- **Mean** averages ALL pixels including background → biased estimate.
  Example: car at depth 0.8 with 30% background at 0.2 → mean = 0.62 (wrong).

- **Median** returns the 50th percentile → robust to background outliers.
  Same example → median ≈ 0.8 (correct, assuming >50% of box is the object).

We additionally crop to the inner 60% of the box before computing the
median, further reducing edge contamination.

### FPS impact of depth estimation

Depth estimation is computationally expensive — it runs a full neural network
(encoder-decoder with attention) on every frame:

| Model variant | CPU time/frame | GPU time/frame | Quality |
|--------------|---------------|----------------|---------|
| `midas_small` | ~30–60 ms | ~5–10 ms | Good for driving |
| `midas_hybrid` | ~80–150 ms | ~10–20 ms | Better edges |
| `midas_large` | ~200–400 ms | ~15–30 ms | Best, but slow |

**Running at lower internal resolution** (`input_resolution: 256` vs 384/512)
is the primary speed optimization:

- The model processes a 256×256 tensor regardless of the 1280×720 display
  frame. The result is upscaled back to frame size via bicubic interpolation.
- At 256, you lose fine-grained depth boundaries (edges blur slightly) but
  retain accurate ordinal depth ordering — which is all that matters for
  per-object distance estimates.
- This is a reasonable and well-documented trick used in production depth
  systems. Quality impact is minimal for our use case (we only need the
  median depth within large bounding boxes, not pixel-perfect boundaries).

---

## Configuration Reference (`configs/config.yaml`)

| Key | Type | Description |
|-----|------|-------------|
| `source.type` | `file/webcam` | Input source selector |
| `source.file_path` | string | Path to video file |
| `source.webcam_index` | int | OS device index for webcam |
| `display.width/height` | int/null | Output window size |
| `display.window_title` | string | OpenCV window title |
| `detection.model_path` | string | YOLO weights file (auto-downloaded) |
| `detection.confidence_threshold` | float | Min detection confidence (0–1) |
| `detection.classes` | list | COCO classes to detect |
| `detection.device` | string | `auto`, `cpu`, or `cuda` |
| `detection.tracker.lost_track_buffer` | int | Frames before a track is deleted |
| `detection.tracker.frame_rate` | int | Source FPS for Kalman motion model |
| `lanes.canny_low_threshold` | int | Canny edge low threshold |
| `lanes.canny_high_threshold` | int | Canny edge high threshold |
| `lanes.smoothing_frames` | int | Frames to average lane lines over |
| `depth.model_name` | string | `midas_small`, `midas_hybrid`, or `midas_large` |
| `depth.input_resolution` | int | Internal inference size (256 recommended for CPU) |
| `depth.device` | string | `auto`, `cpu`, or `cuda` |
| `depth.calibration_scale` | float | Heuristic scale for pseudo-metric conversion |
| `depth.show_heatmap_overlay` | bool | Toggle PIP depth heatmap on/off |
| `logging.level` | string | DEBUG/INFO/WARNING/ERROR |
| `logging.log_file` | string/null | Log file path (null = console only) |
