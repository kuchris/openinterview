"""Process-wide PortAudio reinitialisation.

PortAudio caches the device list when it initialises. To see a device that was unplugged
and plugged back (AirPods, USB mic, BlackHole after reinstall), the library must be
terminated and re-initialised, which invalidates *every* open stream in the process.
With the mic and BlackHole both on PortAudio, one source's reconnect would kill the other.
This module rate-limits resets so all sounddevice-based sources share one reset; each
source's watchdog then sees its stream inactive and reopens it.
"""

from __future__ import annotations

import threading
import time

import sounddevice as sd

_lock = threading.Lock()
_last_reset = 0.0
generation = 0  # incremented on every real reset; sources can compare to detect one


def reset(min_interval_s: float = 2.0) -> bool:
    """Terminate + re-initialise PortAudio unless another source just did. Returns True if reset."""
    global _last_reset, generation
    with _lock:
        now = time.monotonic()
        if now - _last_reset < min_interval_s:
            return False
        sd._terminate()
        sd._initialize()
        _last_reset = now
        generation += 1
        return True


def lock() -> threading.Lock:
    """Hold while opening a stream so it cannot race a concurrent reset."""
    return _lock
