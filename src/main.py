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
    logger.info("AI Driving Assistant — Day 8: Full pipeline with resize optimisation")
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

    # ── Day 8 addition — Stage 0: pre-resize ─────────────────────────
    # Resize every frame DOWN to the pipeline working resolution before
    # any compute-heavy stage sees it.  At 4K input, Canny+Hough alone
    # cost ~180 ms/frame (49% of total time).  Resizing to 1280×720 first
    # drops that to ~15 ms and also fixes the lane-detection threshold
    # mismatch (thresholds were calibrated for ~720p, not 4K).
    pipeline_cfg = config.get("pipeline", {})
    resize_w = pipeline_cfg.get("resize_width", 1280)
    resize_h = pipeline_cfg.get("resize_height", 720)
    if resize_w and resize_h:
        from src.pipeline.resize_stage import FrameResizeStage
        processor.add_stage("resize", FrameResizeStage(width=resize_w, height=resize_h))

    # ── Day 2 addition ────────────────────────────────────────────────
    if "detection" in config:
        from src.detection.stage import DetectionStage
        processor.add_stage("detection", DetectionStage(config["detection"]))

    # ── Day 3 addition ───────────────────────────────────────────────
    if "lanes" in config:
        from src.lanes.stage import LaneDetectionStage
        # Inject visualization toggle so LaneDetectionStage can honour
        # the demo-mode show_lane_overlay flag from the visualization section.
        lanes_cfg = dict(config["lanes"])
        lanes_cfg.setdefault(
            "show_lane_overlay",
            config.get("visualization", {}).get("show_lane_overlay", True),
        )
        processor.add_stage("lanes", LaneDetectionStage(lanes_cfg))

    # ── Day 4 addition ───────────────────────────────────────────────
    # ⚠️  ORDER MATTERS: DepthEstimationStage MUST run AFTER DetectionStage
    # because it reads meta["tracked_objects"] to compute per-object distance
    # estimates. Moving it before detection will produce zero distances.
    if "depth" in config:
        from src.depth.stage import DepthEstimationStage
        # visualization.show_depth_panel maps to depth.show_heatmap_overlay.
        # We override so the demo-mode toggle works without editing depth/stage.py.
        depth_cfg = dict(config["depth"])
        depth_cfg["show_heatmap_overlay"] = config.get("visualization", {}).get(
            "show_depth_panel", depth_cfg.get("show_heatmap_overlay", True)
        )
        processor.add_stage("depth", DepthEstimationStage(depth_cfg))

    # ── Day 5 addition ───────────────────────────────────────────────
    # ⚠️  FULL DEPENDENCY CHAIN — CollisionFusionStage is the FIRST stage
    # that depends on ALL previous stages' outputs:
    #   1. DetectionStage       → meta["tracked_objects"] (track IDs, boxes)
    #   2. LaneDetectionStage   → meta["lane_lines"]      (ego lane bounds)
    #   3. DepthEstimationStage → obj.estimated_distance_m (per-object dist)
    #   4. CollisionFusionStage → fuses 1+2+3 over time → risk annotations
    # This stage MUST be registered LAST. Moving it earlier will produce
    # missing data and incorrect/absent risk estimates.
    if "fusion" in config:
        from src.fusion.stage import CollisionFusionStage
        processor.add_stage("fusion", CollisionFusionStage(config["fusion"]))

    # ── Day 6 addition ───────────────────────────────────────────────
    # ⚠️  AlertStage MUST run LAST in the pipeline. It reads the complete
    # risk annotations from CollisionFusionStage and uses them to:
    #   - Evaluate the debounced alert state machine
    #   - Draw the warning banner and polished context-sensitive labels
    #   - Write meta["active_alert"] and meta["viz_config"] for draw_hud()
    if "alerts" in config:
        from src.alerts.stage import AlertStage
        processor.add_stage(
            "alerts",
            AlertStage(
                config=config["alerts"],
                viz_config=config.get("visualization", {}),
            ),
        )
    # ─────────────────────────────────────────────────────────────────

    # ── Stage list confirmation ───────────────────────────────────────
    # Print every registered stage name + position so we can confirm
    # the full pipeline is wired correctly before the first frame runs.
    logger.info("Pipeline stages registered (%d total):", len(processor.stage_names))
    for i, name in enumerate(processor.stage_names, start=1):
        logger.info("  %d. %s", i, name)

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
