"""
main.py — Entry point for the AI Driving Assistant pipeline.

Wires together: config → logging → VideoSource → FrameProcessor → display loop.

Run from the project root:
    python src/main.py

To switch between a video file and webcam, edit configs/config.yaml —
no code changes required.

Day 1: Raw video loop with FPS overlay.
Day 2: Added DetectionStage (one processor.add_stage() call below).
"""

import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: resolve project root so imports work regardless of CWD
# ---------------------------------------------------------------------------
import sys

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.logger import get_logger, setup_logging
from src.pipeline.video_source import VideoSource
from src.pipeline.frame_processor import FrameProcessor
from src.visualization.display import create_window, destroy_windows, show_frame


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load configuration (validates early so bad config fails fast)
    # ------------------------------------------------------------------
    config_path = _PROJECT_ROOT / "configs" / "config.yaml"
    config = load_config(str(config_path))

    # ------------------------------------------------------------------
    # 2. Set up logging (must happen before any module logs anything)
    # ------------------------------------------------------------------
    log_cfg = config.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("log_file"),
    )
    logger = get_logger(__name__)
    logger.info("=" * 60)
    logger.info("AI Driving Assistant — Day 4: Detection + Tracking + Lanes + Depth")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 3. Initialize the video source
    # ------------------------------------------------------------------
    src_cfg = config["source"]
    source = VideoSource(
        source_type=src_cfg["type"],
        file_path=src_cfg.get("file_path"),
        webcam_index=src_cfg.get("webcam_index", 0),
    )
    meta = source.metadata

    # ------------------------------------------------------------------
    # 4. Initialize the frame processor and register stages
    # ------------------------------------------------------------------
    processor = FrameProcessor()

    # ── Day 2 addition (the ONLY change from Day 1 main.py) ──────────
    # Import and register the detection + tracking stage.
    # Future days will add more stages below this line — main.py never
    # needs structural changes, only new add_stage() calls.
    if "detection" in config:
        from src.detection.stage import DetectionStage
        processor.add_stage("detection", DetectionStage(config["detection"]))

    # ── Day 3 addition ───────────────────────────────────────────────
    if "lanes" in config:
        from src.lanes.stage import LaneDetectionStage
        processor.add_stage("lanes", LaneDetectionStage(config["lanes"]))

    # ── Day 4 addition ───────────────────────────────────────────────
    # ⚠️  ORDER MATTERS: DepthEstimationStage MUST run AFTER DetectionStage
    # because it reads meta["tracked_objects"] to compute per-object distance
    # estimates. Moving it before detection will produce zero distances.
    if "depth" in config:
        from src.depth.stage import DepthEstimationStage
        processor.add_stage("depth", DepthEstimationStage(config["depth"]))
    # ─────────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    # 5. Prepare the display window
    # ------------------------------------------------------------------
    disp_cfg = config["display"]
    window_title: str = disp_cfg["window_title"]
    target_width: int | None = disp_cfg.get("width")
    target_height: int | None = disp_cfg.get("height")

    create_window(window_title)

    # ------------------------------------------------------------------
    # 6. Main frame loop
    # ------------------------------------------------------------------
    logger.info("Entering main loop. Press 'q' or close the window to exit.")

    frame_count = 0
    fps_display = 0.0
    frame_meta: dict = {}

    loop_start_time = time.perf_counter()
    fps_timer_start = loop_start_time

    running = True
    while running:
        frame = source.get_frame()
        if frame is None:
            logger.info("Stream ended — exiting loop.")
            break

        frame_count += 1

        # Pass frame through the processing pipeline
        processed_frame, frame_meta = processor.process(frame)

        # Update FPS estimate every 15 frames to keep it stable
        if frame_count % 15 == 0:
            elapsed = time.perf_counter() - fps_timer_start
            fps_display = 15.0 / elapsed if elapsed > 0 else 0.0
            fps_timer_start = time.perf_counter()

            if fps_display < meta.fps * 0.7:
                logger.warning(
                    "FPS drop detected: %.1f fps (source: %.1f fps)", fps_display, meta.fps
                )

        running = show_frame(
            frame=processed_frame,
            window_title=window_title,
            fps=fps_display,
            meta=frame_meta,
            target_width=target_width,
            target_height=target_height,
        )

    # ------------------------------------------------------------------
    # 7. Teardown and summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - loop_start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0.0

    logger.info("-" * 60)
    logger.info("Pipeline finished.")
    logger.info("  Total frames processed : %d", frame_count)
    logger.info("  Total elapsed time     : %.2f s", total_elapsed)
    logger.info("  Average FPS            : %.2f", avg_fps)
    logger.info("-" * 60)

    source.release()
    destroy_windows()


if __name__ == "__main__":
    main()
