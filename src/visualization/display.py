"""
display.py — Frame rendering, HUD, and all visual overlays.

Visual design (Day 3 polish):
  - Top HUD bar: FPS + object counts + lane status in one semi-transparent strip.
  - Bounding boxes: single-pass glow blend, then sharp outline + minimal label.
  - Lane overlay: translucent fill, bright lines, offset gauge in HUD.
  - No floating text anywhere on the video — everything lives in the HUD.
"""

from typing import TYPE_CHECKING, Any, Optional

import cv2
import numpy as np

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.detection.stage import TrackedObject
    from src.lanes.lane_detector import LaneResult

logger = get_logger(__name__)

# ── Fonts ────────────────────────────────────────────────────────────────────
_FONT_HUD   = cv2.FONT_HERSHEY_DUPLEX    # slightly heavier, better for panels
_FONT_LABEL = cv2.FONT_HERSHEY_SIMPLEX

# ── HUD ──────────────────────────────────────────────────────────────────────
_HUD_H          = 52                      # height of top bar in pixels
_HUD_BG         = (18, 18, 18)           # near-black background
_HUD_BG_ALPHA   = 0.82                   # opacity of HUD panel
_HUD_DIVIDER    = (55, 55, 55)           # subtle separator line color
_HUD_TEXT_COLOR = (220, 220, 220)        # off-white text
_HUD_FONT_SCALE = 0.52
_HUD_THICKNESS  = 1

# ── Detection boxes ──────────────────────────────────────────────────────────
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car":        (220, 100,   0),
    "truck":      (180,  40,   0),
    "bus":        (200,   0, 200),
    "motorcycle": (  0, 200, 255),
    "bicycle":    (  0, 230, 120),
    "person":     (  0,  60, 255),
}
_DEFAULT_COLOR  = (160, 160, 160)
_BOX_THICKNESS  = 2
_GLOW_THICKNESS = 8
_GLOW_ALPHA     = 0.38
_LABEL_SCALE    = 0.48
_LABEL_PAD      = 4

# ── Lane overlay ──────────────────────────────────────────────────────────────
_LANE_COLOR     = (0, 220, 80)
_LANE_FILL      = (0, 200, 60)
_LANE_WIDTH     = 4
_LANE_ALPHA     = 0.28

# ── Lane offset gauge (inside HUD) ───────────────────────────────────────────
_GAUGE_W        = 240    # width of the gauge bar in pixels
_GAUGE_H        = 6      # height of the gauge track
_GAUGE_Y_OFFSET = 40     # y position within the HUD bar
_OFFSET_SAFE    = (0,  210,  70)
_OFFSET_WARN    = (0,  200, 255)
_OFFSET_DANGER  = (0,   40, 255)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def create_window(window_title: str) -> None:
    """Pre-create a resizable OpenCV window before the main loop."""
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    logger.debug("Window '%s' created.", window_title)


def destroy_windows() -> None:
    """Close all OpenCV windows cleanly."""
    cv2.destroyAllWindows()


def show_frame(
    frame: np.ndarray,
    window_title: str,
    fps: float,
    meta: Optional[dict[str, Any]] = None,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> bool:
    """
    Resize, draw HUD, and display the frame.

    Returns False when the user presses 'q' or closes the window.
    """
    display_frame = _resize(frame, target_width, target_height)
    draw_hud(display_frame, fps, meta or {})
    cv2.imshow(window_title, display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        logger.info("Exit key 'q' pressed.")
        return False
    try:
        if cv2.getWindowProperty(window_title, cv2.WND_PROP_VISIBLE) < 1:
            logger.info("Window closed by user.")
            return False
    except cv2.error:
        return False
    return True


def draw_hud(
    frame: np.ndarray,
    fps: float,
    meta: dict[str, Any],
) -> None:
    """
    Draw a semi-transparent top panel containing:
      - FPS counter (left)
      - Object count summary (center)
      - Lane status (right)
      - Offset gauge bar (below text)

    Modifies frame in-place.
    """
    h, w = frame.shape[:2]

    # ── Background panel ─────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, _HUD_H), _HUD_BG, cv2.FILLED)
    cv2.addWeighted(overlay, _HUD_BG_ALPHA, frame, 1 - _HUD_BG_ALPHA, 0, frame)

    # Thin separator line
    cv2.line(frame, (0, _HUD_H), (w, _HUD_H), _HUD_DIVIDER, 1)

    ty = 28   # text baseline y inside the HUD

    # ── FPS (left) ───────────────────────────────────────────────────────────
    fps_color = _OFFSET_SAFE if fps >= 15 else _OFFSET_WARN if fps >= 8 else _OFFSET_DANGER
    _hud_text(frame, f"FPS  {fps:.1f}", (14, ty), fps_color)

    # ── Object counts (center) ───────────────────────────────────────────────
    tracked = meta.get("tracked_objects", [])
    if tracked:
        counts: dict[str, int] = {}
        for obj in tracked:
            counts[obj.class_name] = counts.get(obj.class_name, 0) + 1
        count_str = "   ".join(
            f"{name.upper()} {cnt}" for name, cnt in sorted(counts.items())
        )
    else:
        count_str = "no objects"

    (cw, _), _ = cv2.getTextSize(count_str, _FONT_HUD, _HUD_FONT_SCALE, _HUD_THICKNESS)
    _hud_text(frame, count_str, (w // 2 - cw // 2, ty), _HUD_TEXT_COLOR)

    # ── Lane status (right) ──────────────────────────────────────────────────
    lane_info = meta.get("lane_offset", {})
    if lane_info:
        offset = lane_info.get("normalized", 0.0)
        if abs(offset) < 0.2:
            lane_color, lane_label = _OFFSET_SAFE,   "CENTERED"
        elif abs(offset) < 0.5:
            lane_color, lane_label = _OFFSET_WARN,   "DRIFTING"
        else:
            lane_color, lane_label = _OFFSET_DANGER, "DANGER"

        status_str = f"LANE  {lane_label}"
        (sw, _), _ = cv2.getTextSize(status_str, _FONT_HUD, _HUD_FONT_SCALE, _HUD_THICKNESS)
        _hud_text(frame, status_str, (w - sw - 14, ty), lane_color)

        # ── Offset gauge bar ─────────────────────────────────────────────────
        gx = w // 2 - _GAUGE_W // 2
        gy = _GAUGE_Y_OFFSET
        # Track
        cv2.rectangle(frame, (gx, gy), (gx + _GAUGE_W, gy + _GAUGE_H), (50, 50, 50), cv2.FILLED)
        # Center mark
        cx = gx + _GAUGE_W // 2
        cv2.line(frame, (cx, gy - 2), (cx, gy + _GAUGE_H + 2), (90, 90, 90), 1)
        # Marker position
        marker_x = int(cx + offset * (_GAUGE_W // 2))
        marker_x = max(gx + 4, min(gx + _GAUGE_W - 4, marker_x))
        cv2.circle(frame, (marker_x, gy + _GAUGE_H // 2), 7, lane_color, cv2.FILLED)
        cv2.circle(frame, (marker_x, gy + _GAUGE_H // 2), 7, (255, 255, 255), 1)


def draw_detections(
    frame: np.ndarray,
    tracked_objects: "list[TrackedObject]",
) -> np.ndarray:
    """
    Draw glowing bounding boxes with minimal labels.

    Two-pass approach: one addWeighted call blends all glow layers at once,
    then sharp outlines and labels are drawn on top. This avoids N separate
    blending operations per frame.

    Label format: "Car #7"  (class + track ID only — no confidence clutter)
    """
    if not tracked_objects:
        return frame

    # Pass 1 — glow layer (single blend)
    glow = frame.copy()
    for obj in tracked_objects:
        color = _CLASS_COLORS.get(obj.class_name.lower(), _DEFAULT_COLOR)
        cv2.rectangle(
            glow,
            (obj.x1 - 2, obj.y1 - 2),
            (obj.x2 + 2, obj.y2 + 2),
            color, _GLOW_THICKNESS,
        )
    cv2.addWeighted(glow, _GLOW_ALPHA, frame, 1 - _GLOW_ALPHA, 0, frame)

    # Pass 2 — sharp boxes + labels
    for obj in tracked_objects:
        color = _CLASS_COLORS.get(obj.class_name.lower(), _DEFAULT_COLOR)

        cv2.rectangle(frame, (obj.x1, obj.y1), (obj.x2, obj.y2), color, _BOX_THICKNESS)

        label = f"{obj.class_name.capitalize()} #{obj.track_id}"
        (lw, lh), bl = cv2.getTextSize(label, _FONT_LABEL, _LABEL_SCALE, 1)

        lx = obj.x1
        ly = max(obj.y1 - _LABEL_PAD, lh + _LABEL_PAD * 2)

        cv2.rectangle(
            frame,
            (lx, ly - lh - _LABEL_PAD),
            (lx + lw + _LABEL_PAD * 2, ly + bl + _LABEL_PAD - 2),
            color, cv2.FILLED,
        )
        cv2.putText(
            frame, label,
            (lx + _LABEL_PAD, ly),
            _FONT_LABEL, _LABEL_SCALE,
            (255, 255, 255), 1, cv2.LINE_AA,
        )

    return frame


def draw_lane_overlay(
    frame: np.ndarray,
    lane_result: "LaneResult",
) -> np.ndarray:
    """
    Draw translucent lane fill + bright lane lines.

    The offset indicator is handled by draw_hud() — nothing is drawn
    as floating text on the video here.
    """
    left  = lane_result.left_line
    right = lane_result.right_line

    # Translucent fill
    if left and right:
        overlay = frame.copy()
        pts = np.array([
            [left[2],  left[3]],
            [right[2], right[3]],
            [right[0], right[1]],
            [left[0],  left[1]],
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], _LANE_FILL)
        cv2.addWeighted(overlay, _LANE_ALPHA, frame, 1 - _LANE_ALPHA, 0, frame)

    # Lane lines
    if left:
        cv2.line(frame, (left[0], left[1]), (left[2], left[3]),
                 _LANE_COLOR, _LANE_WIDTH, cv2.LINE_AA)
    if right:
        cv2.line(frame, (right[0], right[1]), (right[2], right[3]),
                 _LANE_COLOR, _LANE_WIDTH, cv2.LINE_AA)

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resize(
    frame: np.ndarray,
    width: Optional[int],
    height: Optional[int],
) -> np.ndarray:
    if width is None or height is None:
        return frame
    sh, sw = frame.shape[:2]
    if sw == width and sh == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def _hud_text(
    frame: np.ndarray,
    text: str,
    pos: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    """Draw drop-shadowed HUD text at pos."""
    x, y = pos
    cv2.putText(frame, text, (x + 1, y + 1),
                _FONT_HUD, _HUD_FONT_SCALE, (0, 0, 0), _HUD_THICKNESS + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y),
                _FONT_HUD, _HUD_FONT_SCALE, color, _HUD_THICKNESS, cv2.LINE_AA)
