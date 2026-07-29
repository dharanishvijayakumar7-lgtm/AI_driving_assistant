"""
debug_bugs2.py - Unicode-safe diagnostic
"""
import sys, pathlib, warnings
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from src.utils.logger import setup_logging
setup_logging(level="WARNING")

from src.utils.config import load_config
from src.pipeline.video_source import VideoSource
import cv2, numpy as np

config = load_config("configs/config.yaml")
src_cfg = config["source"]
source = VideoSource(source_type=src_cfg["type"],
                     file_path=src_cfg.get("file_path"),
                     webcam_index=src_cfg.get("webcam_index", 0))

frame_raw = source.get_frame()
source.release()
W, H = 1280, 720
frame = cv2.resize(frame_raw, (W, H))

# ============================================================
# BUG 1
# ============================================================
print("=" * 60)
print("BUG 1: Lane line geometry")
print("=" * 60)
from src.lanes.lane_detector import LaneDetector
result = LaneDetector(dict(config["lanes"])).detect(frame)

L = result.left_line   # (x_bot, y_bot, x_top, y_top)
R = result.right_line
print(f"left_line  = {L}")
print(f"right_line = {R}")
print(f"Frame W={W} H={H}")
print()

if L:
    print(f"LEFT  bottom=({L[0]},{L[1]})  top=({L[2]},{L[3]})")
    print(f"  x_bottom {L[0]} -> {'LEFT half OK' if L[0] < W//2 else 'RIGHT half WRONG'}")
    print(f"  x_top    {L[2]} -> should converge toward center vanishing point")
if R:
    print(f"RIGHT bottom=({R[0]},{R[1]})  top=({R[2]},{R[3]})")
    print(f"  x_bottom {R[0]} -> {'RIGHT half OK' if R[0] > W//2 else 'LEFT half WRONG'}")
    print(f"  x_top    {R[2]} -> should converge toward center vanishing point")

print()
print("-- Polygon shape analysis --")
if L and R:
    # The polygon is drawn as: left_top -> right_top -> right_bot -> left_bot
    pts = [(L[2],L[3]), (R[2],R[3]), (R[0],R[1]), (L[0],L[1])]
    print(f"Polygon: {pts}")
    # For a proper lane corridor: left_x < right_x at BOTH top and bottom
    top_ok = L[2] < R[2]
    bot_ok = L[0] < R[0]
    print(f"At top: left_x={L[2]} < right_x={R[2]} -> {'OK' if top_ok else 'CROSSED - BUG!'}")
    print(f"At bot: left_x={L[0]} < right_x={R[0]} -> {'OK' if bot_ok else 'CROSSED - BUG!'}")
    print()
    print("DIAGNOSIS:")
    if not top_ok:
        print("  TOP POINTS SWAPPED -> X shape confirmed")
        print("  Root: LaneDetector stores (x_bottom, y_bottom, x_top, y_top) where")
        print("  'left' top (vanishing pt) is at x=896 (right of center)")
        print("  and 'right' top is at x=741 (left of center)")
        print("  -> Lines cross at the horizon -> X shape.")
        print()
        print("  Fix: the lane lines DO correctly run from their respective sides")
        print("  at the bottom to a COMMON vanishing point near center.")
        print("  The 'X' is just perspective convergence drawn incorrectly.")
        print("  The real fix: truncate lines so they don't cross, or swap top x values.")

print()
print("  Line slopes in image coords:")
if L:
    slope_L = (L[3]-L[1]) / (L[2]-L[0]) if (L[2]-L[0]) != 0 else float('inf')
    print(f"  LEFT  slope = (y_top-y_bot)/(x_top-x_bot) = ({L[3]}-{L[1]})/({L[2]}-{L[0]}) = {slope_L:.3f}")
if R:
    slope_R = (R[3]-R[1]) / (R[2]-R[0]) if (R[2]-R[0]) != 0 else float('inf')
    print(f"  RIGHT slope = (y_top-y_bot)/(x_top-x_bot) = ({R[3]}-{R[1]})/({R[2]}-{R[0]}) = {slope_R:.3f}")

# ============================================================
# BUG 2
# ============================================================
print()
print("=" * 60)
print("BUG 2: Depth map orientation audit")
print("=" * 60)

from src.depth.depth_estimator import DepthEstimator
from src.depth.depth_utils import colorize_depth_map

depth_cfg = dict(config["depth"])
estimator = DepthEstimator(depth_cfg)
depth_map = estimator.estimate(frame)
print(f"depth_map shape={depth_map.shape}  dtype={depth_map.dtype}")
print(f"depth_map range: [{depth_map.min():.4f}, {depth_map.max():.4f}]  mean={depth_map.mean():.4f}")

# MiDaS: higher = closer. Sky should be low (far), road should be mid-high
# But MiDaS also sometimes inverts. Check top vs bottom.
sky_mean  = depth_map[:H//5, :].mean()
road_mean = depth_map[4*H//5:, :].mean()
print(f"sky rows    (top 20%): depth mean = {sky_mean:.4f}")
print(f"road rows   (bot 20%): depth mean = {road_mean:.4f}")
print(f"Road > Sky (expected for MiDaS inverse-depth): {road_mean > sky_mean}")

depth_colored = colorize_depth_map(depth_map)
side_by_side = np.hstack([frame, cv2.resize(depth_colored, (W, H))])
cv2.imwrite("debug_depth_comparison.jpg", side_by_side)
print(f"Saved: debug_depth_comparison.jpg")

# ============================================================
# BUG 3
# ============================================================
print()
print("=" * 60)
print("BUG 3: Distance sign / value audit")
print("=" * 60)

from src.depth.depth_utils import relative_to_pseudo_meters
cal = depth_cfg.get("calibration_scale", 30.0)
print(f"calibration_scale = {cal}")
print(f"Formula: pseudo_distance = {cal} / (relative_depth + 1e-4)")
print()
print(f"{'rel_depth':>12}  {'pseudo_m':>10}  sign_ok")
for d in [0.9, 0.7, 0.5, 0.3, 0.1, 0.01, 0.001]:
    m = relative_to_pseudo_meters(d, cal)
    print(f"{d:>12.4f}  {m:>10.2f}  {'OK' if m > 0 else 'NEGATIVE BUG'}")

# Now check what the stage ACTUALLY assigns
print()
print("Checking DepthEstimationStage distance assignment code:")
src = pathlib.Path("src/depth/stage.py").read_text()
# Find the block around estimated_distance_m
idx = src.find("obj.estimated_distance_m")
snippet = src[max(0,idx-400):idx+200]
print(snippet)
print("=" * 60)

# Draw a lane debug frame to visualize the X
debug_frame = frame.copy()
if L:
    cv2.line(debug_frame, (L[0], L[1]), (L[2], L[3]), (0, 255, 0), 2)
    cv2.circle(debug_frame, (L[0], L[1]), 8, (0, 0, 255), -1)  # red = bottom
    cv2.circle(debug_frame, (L[2], L[3]), 8, (0, 255, 0), -1)  # green = top
    cv2.putText(debug_frame, "L-bot", (L[0]+5, L[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(debug_frame, "L-top", (L[2]+5, L[3]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
if R:
    cv2.line(debug_frame, (R[0], R[1]), (R[2], R[3]), (0, 255, 255), 2)
    cv2.circle(debug_frame, (R[0], R[1]), 8, (0, 0, 255), -1)
    cv2.circle(debug_frame, (R[2], R[3]), 8, (0, 255, 0), -1)
    cv2.putText(debug_frame, "R-bot", (R[0]+5, R[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
    cv2.putText(debug_frame, "R-top", (R[2]+5, R[3]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
cv2.imwrite("debug_lane_lines.jpg", debug_frame)
print("Saved: debug_lane_lines.jpg")
