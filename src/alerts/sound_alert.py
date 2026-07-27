"""
sound_alert.py — Programmatically generated beep alert for DANGER events.

Design choices
--------------
* **No external audio files**: The beep tone is synthesized with numpy
  (a sine wave at 880 Hz for 0.25 s), then played via sounddevice.  This
  avoids shipping a .wav/.mp3 asset and keeps the dependency footprint tiny.
* **Fallback-safe**: If sounddevice is unavailable (headless server, CI),
  the play function logs a warning and returns silently — the rest of the
  pipeline keeps running. Audio is a nice-to-have, not a hard requirement.
* **Play-once, not every frame**: The caller (AlertStage) is responsible for
  only calling `play_beep()` on state *transitions* (SAFE→DANGER), not
  continuously while the alert is active. This module has no internal state —
  it is a pure function that blocks briefly to play the tone.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)

# ── Tone parameters ───────────────────────────────────────────────────────────
_SAMPLE_RATE = 22050   # Hz — half CD quality, plenty for a beep
_FREQUENCY   = 880     # Hz — A5 note, attention-grabbing but not harsh
_DURATION    = 0.25    # seconds
_AMPLITUDE   = 0.4     # 0.0–1.0 volume (lower to avoid startling)


def _generate_beep() -> np.ndarray:
    """Generate a short sine-wave beep as a float32 numpy array."""
    n_samples = int(_SAMPLE_RATE * _DURATION)
    t = np.linspace(0.0, _DURATION, n_samples, endpoint=False, dtype=np.float32)
    # Sine wave with a short linear fade-out to avoid the click artefact
    # at the end of an abruptly cut waveform.
    tone = _AMPLITUDE * np.sin(2.0 * math.pi * _FREQUENCY * t)
    fade_samples = int(n_samples * 0.1)
    if fade_samples > 0:
        fade = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        tone[-fade_samples:] *= fade
    return tone


def play_beep() -> None:
    """
    Play a short attention beep synchronously.

    This call blocks for approximately ``_DURATION`` seconds while the tone
    plays. For a 0.25 s beep this is imperceptible relative to normal frame
    processing latency. If you need non-blocking audio, run this in a
    daemon thread, but for our use case the brief block is acceptable.

    Gracefully degrades (logs warning, returns) if sounddevice is not
    installed or the audio device is unavailable.
    """
    try:
        import sounddevice as sd  # type: ignore
        tone = _generate_beep()
        sd.play(tone, samplerate=_SAMPLE_RATE)
        sd.wait()
    except ImportError:
        logger.warning(
            "sounddevice not installed — audio alerts disabled. "
            "Install with: pip install sounddevice"
        )
    except Exception as exc:  # noqa: BLE001
        # Catch OSError (no audio device), PortAudioError, etc.
        logger.warning("Audio playback failed: %s", exc)
