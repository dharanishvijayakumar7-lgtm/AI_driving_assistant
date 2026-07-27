"""
alert_manager.py — Priority, debounce, and hysteresis for driver-facing alerts.

Why debounce/hysteresis matters for safety alerts
--------------------------------------------------
A raw threshold check fires an alert the moment TTC < 2.0 s and clears it the
moment TTC ≥ 2.0 s. In practice, monocular depth estimates are noisy enough
that TTC might oscillate between 1.9 s and 2.1 s on consecutive frames — even
for an object that is steadily approaching. Without debouncing, this produces
a warning banner that flashes on/off every other frame.

Flickering warnings are *worse* than no warning at all:

1. **Alarm fatigue**: A driver who sees the warning flicker 20 times in 5
   seconds starts ignoring it. When a real threat triggers a sustained warning,
   they no longer respond appropriately.

2. **Trust degradation**: A system that cries wolf is eventually disabled by
   the operator. This is the #1 cause of safety feature bypasses in production
   ADAS systems (see Mobileye's "nuisance rate" metric — they target < 1 false
   alarm per 10 000 km).

3. **Cognitive load**: Flickering is visually distracting and increases driver
   reaction time to *real* events, the opposite of the intended effect.

The solution: hysteresis with two separate counters.
  - ``danger_persist_frames`` (e.g., 5): A DANGER condition must hold for this
    many *consecutive* frames before the alert activates. This filters out
    single-frame noise spikes.
  - ``clear_persist_frames`` (e.g., 10): Once active, the alert stays on for
    this many consecutive frames of clear readings before dismissing. This
    prevents the alert from vanishing instantly when TTC briefly bounces back
    above the threshold.

The asymmetry (5 to trigger, 10 to clear) is intentional: it is safer to
err on the side of keeping the warning visible slightly longer after the
immediate threat recedes — giving the driver time to confirm the situation
has actually resolved.

Real-world parallel: industrial PLC safety interlocks use the same two-counter
hysteresis. ABS systems use a similar principle for wheel-slip detection.

Priority selection
------------------
When multiple in-lane objects are simultaneously in DANGER state, we display
exactly ONE alert — the most urgent one (lowest TTC). Displaying 3 overlapping
"DANGER" banners simultaneously would be more confusing than helpful. The
single most urgent threat is what the driver needs to react to first.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ── Alert data contract ───────────────────────────────────────────────────────

@dataclass
class ActiveAlert:
    """
    Represents a currently active driver-facing warning.

    Attributes
    ----------
    track_id : int
        ByteTrack ID of the threatening object.
    message : str
        Human-readable warning text for the banner
        (e.g., "⚠ COLLISION RISK — Car #7, 1.4s").
    severity : str
        "DANGER" or "CAUTION".
    ttc_seconds : float or None
        Current TTC for display in the banner.
    triggered_at : float
        time.perf_counter() timestamp when the alert first triggered.
    seconds_active : float
        How long this alert has been continuously active (updated each frame).
    """
    track_id: int
    message: str
    severity: str
    ttc_seconds: Optional[float]
    triggered_at: float
    seconds_active: float = field(default=0.0)


# ── Alert manager ─────────────────────────────────────────────────────────────

class AlertManager:
    """
    Evaluates tracked objects and manages the lifecycle of driver-facing alerts.

    Implements a two-counter hysteresis state machine:

        [NO ALERT]  ─ DANGER persists N frames ──►  [PENDING]
        [PENDING]   ─ condition clears         ──►  [NO ALERT]  (never triggered)
        [PENDING]   ─ N frames elapsed         ──►  [ALERT ACTIVE]
        [ALERT ACTIVE] ─ clear for M frames    ──►  [NO ALERT]

    Parameters
    ----------
    danger_persist_frames : int
        Consecutive DANGER frames required before triggering the alert.
    clear_persist_frames : int
        Consecutive clear frames required before dismissing the alert.
    sound_enabled : bool
        Whether to play an audio beep on new alert transitions.
    """

    def __init__(
        self,
        danger_persist_frames: int = 5,
        clear_persist_frames: int = 10,
        sound_enabled: bool = False,
    ) -> None:
        self._danger_persist = danger_persist_frames
        self._clear_persist = clear_persist_frames
        self._sound_enabled = sound_enabled

        # Consecutive-frame counters
        self._danger_counter: int = 0   # frames DANGER condition has been present
        self._clear_counter: int = 0    # frames since DANGER condition went away

        # Active alert state
        self._active_alert: Optional[ActiveAlert] = None
        self._alert_is_live: bool = False

        logger.info(
            "AlertManager ready (persist=%d, clear=%d, sound=%s).",
            danger_persist_frames,
            clear_persist_frames,
            sound_enabled,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def evaluate(self, tracked_objects: list[Any]) -> Optional[ActiveAlert]:
        """
        Evaluate all tracked objects and advance the alert state machine.

        Steps:
          1. Filter to in-lane objects only.
          2. From those, find the highest-priority (lowest TTC) DANGER object.
          3. Advance the hysteresis counters.
          4. Return the active alert if one is live, else None.

        Parameters
        ----------
        tracked_objects : list
            TrackedObject instances with risk_level, in_ego_lane, ttc_seconds,
            track_id, class_name attributes (set by CollisionFusionStage).

        Returns
        -------
        ActiveAlert or None
            The current live alert, or None if no alert is active.
        """
        # ── 1. Find the most urgent in-lane DANGER threat ────────────────
        best_threat = self._pick_highest_priority(tracked_objects)

        # ── 2. Advance the hysteresis state machine ───────────────────────
        if best_threat is not None:
            # DANGER condition is present this frame
            self._danger_counter += 1
            self._clear_counter = 0
        else:
            # No DANGER condition this frame
            self._clear_counter += 1
            self._danger_counter = 0

        # ── 3. State transitions ──────────────────────────────────────────
        if not self._alert_is_live:
            # Not yet triggered — check if we've hit the persist threshold
            if self._danger_counter >= self._danger_persist and best_threat is not None:
                self._trigger_alert(best_threat)
        else:
            # Alert is live — check if we should dismiss it
            if self._clear_counter >= self._clear_persist:
                self._dismiss_alert()
            elif best_threat is not None:
                # Update the active alert with the current best threat
                self._update_alert(best_threat)

        # ── 4. Tick seconds_active ────────────────────────────────────────
        if self._active_alert is not None:
            self._active_alert.seconds_active = (
                time.perf_counter() - self._active_alert.triggered_at
            )

        return self._active_alert

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _pick_highest_priority(tracked_objects: list[Any]) -> Optional[Any]:
        """
        Return the in-lane DANGER object with the lowest TTC, or None.

        We only alert on in-lane objects because off-lane threats (even with
        a low TTC) are not in the ego vehicle's path and should not trigger
        a driver-facing alert.
        """
        candidates = [
            obj for obj in tracked_objects
            if getattr(obj, "in_ego_lane", False)
            and getattr(obj, "risk_level", "SAFE") == "DANGER"
            and getattr(obj, "ttc_seconds", None) is not None
        ]
        if not candidates:
            return None
        # Lowest TTC = most urgent threat
        return min(candidates, key=lambda o: o.ttc_seconds)

    def _trigger_alert(self, threat: Any) -> None:
        """Create a new active alert and optionally play sound."""
        now = time.perf_counter()
        ttc = threat.ttc_seconds
        label = f"{threat.class_name.capitalize()} #{threat.track_id}"
        ttc_str = f"{ttc:.1f}s" if ttc is not None else "?"
        message = f"\u26a0  COLLISION RISK \u2014 {label}  |  TTC {ttc_str}"

        self._active_alert = ActiveAlert(
            track_id=threat.track_id,
            message=message,
            severity="DANGER",
            ttc_seconds=ttc,
            triggered_at=now,
        )
        self._alert_is_live = True

        logger.warning(
            "ALERT TRIGGERED: track_id=%d, TTC=%.1fs",
            threat.track_id,
            ttc or -1.0,
        )

        # Play beep only on the initial trigger (not every frame)
        if self._sound_enabled:
            try:
                from src.alerts.sound_alert import play_beep
                import threading
                # Run in a daemon thread so it doesn't block the pipeline
                t = threading.Thread(target=play_beep, daemon=True)
                t.start()
            except Exception as exc:  # noqa: BLE001
                logger.debug("Sound alert failed: %s", exc)

    def _update_alert(self, threat: Any) -> None:
        """Refresh the TTC value of the live alert (same or new threat)."""
        if self._active_alert is None:
            return
        ttc = threat.ttc_seconds
        label = f"{threat.class_name.capitalize()} #{threat.track_id}"
        ttc_str = f"{ttc:.1f}s" if ttc is not None else "?"
        self._active_alert.message = (
            f"\u26a0  COLLISION RISK \u2014 {label}  |  TTC {ttc_str}"
        )
        self._active_alert.track_id = threat.track_id
        self._active_alert.ttc_seconds = ttc

    def _dismiss_alert(self) -> None:
        """Clear the active alert."""
        if self._active_alert is not None:
            logger.info(
                "Alert dismissed after %.1f s.",
                self._active_alert.seconds_active,
            )
        self._active_alert = None
        self._alert_is_live = False
        self._danger_counter = 0
        self._clear_counter = 0
