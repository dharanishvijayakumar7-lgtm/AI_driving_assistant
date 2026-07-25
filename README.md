# AI Driving Assistant 🚗

A real-time computer vision pipeline for autonomous driving assistance.
Built incrementally over 10 days.

| Day | Status | Feature |
|-----|--------|---------|
| ✅ 1 | Done | Project scaffold + video pipeline |
| ✅ 2 | Done | Object detection (YOLOv8n) + Multi-object tracking (ByteTrack) |
| 3   | Next | Lane detection |
| 4   |      | Monocular depth estimation |
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

On first run, `yolov8n.pt` (~6 MB) is downloaded automatically into the
Ultralytics cache. Press **`q`** or close the window to exit.

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

| Hardware | Detector | Detection FPS | Day 1 baseline |
|----------|----------|---------------|----------------|
| **CPU only** (i5/i7, 4-8 cores) | YOLOv8n | **8–15 FPS** | ~60+ FPS |
| **CPU only** | YOLOv8s | 4–8 FPS | ~60+ FPS |
| **NVIDIA GPU** (RTX 3060+) | YOLOv8n | 60–120 FPS | ~200+ FPS |
| **NVIDIA GPU** | YOLOv8s | 45–90 FPS | ~200+ FPS |

> CPU FPS is limited by PyTorch inference, not OpenCV. We'll add quantization
> and frame-skip optimizations on Day 8.

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

### Day 2 — How DetectionStage plugged in

```python
# src/main.py — the ONLY change from Day 1:
from src.detection.stage import DetectionStage
processor.add_stage("detection", DetectionStage(config["detection"]))
```

`video_source.py`, `frame_processor.py`, and the rest of `main.py` are
**completely unchanged** from Day 1.

### Adding the next stage (Day 3 example)

```python
from src.lanes.stage import LaneDetectionStage
processor.add_stage("lanes", LaneDetectionStage(config["lanes"]))
# Can read meta["tracked_objects"] if needed
```

### Meta dict keys (growing each day)

| Stage added   | Key written to meta      | Read by future stages       |
|---------------|--------------------------|-----------------------------|
| Day 2 detect  | `tracked_objects`        | depth, fusion, alerts       |
| Day 3 lanes   | `lanes`                  | fusion, alerts              |
| Day 4 depth   | `depth_map`              | fusion, alerts              |
| Day 5 fusion  | `risk_scores`            | alerts                      |

---

## How ByteTrack Works

ByteTrack solves the problem that standard trackers lose IDs whenever
detector confidence briefly dips (e.g. during occlusion):

1. **High-confidence matching** — Detections above the activation threshold
   are matched to existing tracks via Kalman-filter-predicted positions +
   Hungarian-algorithm IoU matching.

2. **Low-confidence rescue** — Detections that scored too low to start a
   new track (but > 0.1) are matched against tracks that went *unmatched*
   in step 1. This prevents a briefly-occluded car from being declared lost.

3. **Lifecycle** — New tracks start "tentative" and are confirmed after N
   frames. Unmatched confirmed tracks are held "lost" for `lost_track_buffer`
   frames (Kalman keeps predicting their position) then deleted.

**Result:** IDs stay stable through occlusions, lighting changes, and
crowd scenes — the conditions that flip SORT-based trackers every few frames.

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
| `logging.level` | string | DEBUG/INFO/WARNING/ERROR |
| `logging.log_file` | string/null | Log file path (null = console only) |
