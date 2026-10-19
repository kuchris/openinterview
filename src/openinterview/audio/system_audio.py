"""System audio capture: native Core Audio tap helper, with BlackHole as fallback.

Primary path: `bin/sysaudio` (helpers/sysaudio/main.swift) captures every app's output with
a Core Audio Process Tap, downmixes and resamples to 16 kHz natively, and streams framed
float32 PCM on stdout. This module supervises that process:

  * restart with exponential backoff if it exits, errors, or stops sending its 1 s heartbeat.
    The tap delivers *no frames at all* while no app is playing audio, so missing frames are
    normal. When frames resume after such an idle gap, the gap is filled with silence to
    keep the system timeline aligned with wall-clock time (the echo gate relies on that).
  * restart immediately when the helper reports the default output device changed
    (exit code 75), e.g. AirPods connecting, so the tap follows the new device

Fallback path: the BlackHole virtual loopback device, read like a microphone through
PortAudio. Used when the helper cannot run (macOS < 14.2, not built, tap creation fails).
"""

from __future__ import annotations

import json
import logging
import platform
import struct
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT
from .buffer import AudioRingBuffer
from .source import SourceState, StatusCallback

log = logging.getLogger(__name__)

HELPER_PATH = PROJECT_ROOT / "bin" / "sysaudio"
FRAME_MAGIC = 0x31554153  # 'SAU1'
HEADER = struct.Struct("<IIQ")
EXIT_DEVICE_CHANGED = 75
STALL_SECONDS = 5.0          # no heartbeat for this long = hung helper
MAX_GAP_FILL_SECONDS = 10.0  # longer idle gaps are not zero-filled (timeline just jumps)
START_TIMEOUT_SECONDS = 60.0
MAX_BACKOFF_SECONDS = 15.0

BLACKHOLE_SETUP = """\
BlackHole fallback setup:
  1. brew install blackhole-2ch            (or https://existential.audio/blackhole/)
  2. Open Audio MIDI Setup > "+" > Create Multi-Output Device; tick your speakers/headphones
     AND "BlackHole 2ch" (put your real output first, enable drift correction on BlackHole)
  3. Select that Multi-Output Device as the system output (or the call app's speaker)
  4. Run with: openinterview run --source system   (or set audio.system_backend: blackhole)
Note: macOS volume keys do not control a Multi-Output Device; set volume per device."""


def helper_available(path: Path = HELPER_PATH) -> bool:
    return path.is_file() and path.stat().st_mode & 0o111 != 0


def process_tap_supported() -> bool:
    try:
        major, minor = (int(x) for x in platform.mac_ver()[0].split(".")[:2])
    except ValueError:
        return False
    return (major, minor) >= (14, 2)


def helper_permissions(path: Path = HELPER_PATH) -> dict:
    out = subprocess.run([str(path), "permissions"], capture_output=True, text=True, timeout=10)
    return json.loads(out.stdout)


def find_blackhole() -> str | None:
    from .microphone import list_input_devices

    for d in list_input_devices():
        if "blackhole" in d["name"].lower():
            return d["name"]
    return None


def read_frames(stream, on_frame) -> None:
    """Parse helper frames from a binary stream until EOF. Calls on_frame(samples, end_time)."""
    while True:
        header = stream.read(HEADER.size)
        if len(header) < HEADER.size:
            return
        magic, count, end_ns = HEADER.unpack(header)
        if magic != FRAME_MAGIC:
            raise ValueError(f"bad frame magic {magic:#x}")
        payload = stream.read(count * 4)
        if len(payload) < count * 4:
            return
        on_frame(np.frombuffer(payload, dtype="<f4"), end_ns / 1e9)  # zero-copy view


class SystemAudioSource:
    """Supervised `sysaudio` helper process."""

    name = "system"

    def __init__(
        self,
        buffer: AudioRingBuffer,
        helper: Path = HELPER_PATH,
        on_status: StatusCallback | None = None,
        command: list[str] | None = None,  # override for tests
    ) -> None:
        self.buffer = buffer
        self.helper = helper
        self.on_status = on_status
        self.command = command
        self.state = SourceState.STOPPED
        self.frames = 0
        self.nonzero_frames = 0
        self.restarts = 0
        self.device_changes = 0
        self.last_capture_latency = 0.0
        self.last_error = ""
        self.helper_format: dict = {}
        self._proc: subprocess.Popen | None = None
        self._last_heartbeat = 0.0
        self._last_end_time: float | None = None
        self.gap_fill_seconds = 0.0
        self._started_event = threading.Event()
        self._stop = threading.Event()
        self._supervisor: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        if self.command is None and not helper_available(self.helper):
            raise RuntimeError(
                f"system audio helper not found at {self.helper}; run scripts/build_helper.sh"
            )
        self._stop.clear()
        self._supervisor = threading.Thread(target=self._supervise, name="sysaudio-supervisor",
                                            daemon=True)
        self._supervisor.start()

    def wait_started(self, timeout: float) -> bool:
        """True once the helper reported a running tap (used for fallback decisions)."""
        return self._started_event.wait(timeout)

    def stop(self) -> None:
        self._stop.set()
        self._terminate()
        if self._supervisor:
            self._supervisor.join(timeout=5)
        self._set_state(SourceState.STOPPED, "")

    def stats(self) -> dict:
        return {
            "device": "Core Audio process tap",
            "state": self.state.value,
            "frames": self.frames,
            "nonzero_frames": self.nonzero_frames,
            "restarts": self.restarts,
            "idle_gap_filled_s": round(self.gap_fill_seconds, 1),
            "device_changes": self.device_changes,
            "capture_latency_ms": round(self.last_capture_latency * 1000, 1),
            **({"last_error": self.last_error} if self.last_error else {}),
            **({"format": self.helper_format} if self.helper_format else {}),
        }

    # -- supervision -------------------------------------------------------------------

    def _set_state(self, state: SourceState, detail: str) -> None:
        self.state = state
        if self.on_status:
            self.on_status(self.name, state, detail)

    def _launch(self) -> subprocess.Popen:
        cmd = self.command or [str(self.helper), "capture", "--rate", str(self.buffer.sample_rate)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, bufsize=0)
        self._proc = proc
        self._last_heartbeat = time.monotonic()
        for target, name in ((self._read_stdout, "sys-pcm"), (self._read_stderr, "sys-log")):
            threading.Thread(target=target, args=(proc,), name=name, daemon=True).start()
        return proc

    def _terminate(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.close()  # helper exits when stdin closes
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    def _supervise(self) -> None:
        backoff = 1.0
        first = True
        while not self._stop.is_set():
            self._set_state(SourceState.STARTING if first else SourceState.RECONNECTING,
                            "launching helper")
            proc = self._launch()
            first = False
            healthy_since = None
            reason = ""
            launched = time.monotonic()
            hinted = False
            while not self._stop.wait(0.25):
                waiting = time.monotonic() - launched
                if self.state != SourceState.RUNNING:
                    # Starting a tap blocks while macOS shows the permission dialog (first run,
                    # or after a rebuild). Restarting would not help, so wait and say why.
                    if waiting > 5 and not hinted:
                        hinted = True
                        self._set_state(self.state, "waiting for system audio permission: "
                                        "look for a macOS dialog about 'sysaudio'")
                    if waiting > START_TIMEOUT_SECONDS:
                        reason = "helper did not start"
                        self._terminate()
                        break
                code = proc.poll()
                if code is not None:
                    if code == EXIT_DEVICE_CHANGED:
                        self.device_changes += 1
                        reason = "output device changed"
                    else:
                        reason = f"helper exited ({code}){': ' + self.last_error if self.last_error else ''}"
                    break
                if self.state == SourceState.RUNNING:
                    healthy_since = healthy_since or time.monotonic()
                    if time.monotonic() - self._last_heartbeat > STALL_SECONDS:
                        reason = "helper stopped responding"
                        self._terminate()
                        break
            if self._stop.is_set():
                break
            log.warning("system audio: %s; restarting", reason)
            self.restarts += 1
            self._set_state(SourceState.RECONNECTING, reason)
            if reason == "output device changed":
                backoff = 1.0
                continue  # follow the new device right away
            if healthy_since and time.monotonic() - healthy_since > 30:
                backoff = 1.0  # it had been running fine: treat as a fresh failure
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
        self._terminate()

    # -- helper I/O --------------------------------------------------------------------

    def _on_frame(self, samples: np.ndarray, end_time: float) -> None:
        self.frames += 1
        sr = self.buffer.sample_rate
        start_time = end_time - samples.shape[0] / sr
        if self._last_end_time is not None:
            gap = start_time - self._last_end_time
            if 0.05 < gap <= MAX_GAP_FILL_SECONDS:  # idle output: nobody was playing
                self.buffer.write(np.zeros(int(gap * sr), np.float32), start_time)
                self.gap_fill_seconds += gap
        self._last_end_time = end_time
        if samples.any():
            self.nonzero_frames += 1
        self.last_capture_latency = max(0.0, time.monotonic() - end_time)
        self.buffer.write(samples, end_time)

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        try:
            read_frames(proc.stdout, self._on_frame)
        except Exception as exc:
            log.error("system audio stream error: %s", exc)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for raw in proc.stderr:
            try:
                msg = json.loads(raw)
            except ValueError:
                log.debug("sysaudio: %s", raw.strip())
                continue
            event = msg.get("event")
            if event == "format":
                self.helper_format = msg
                log.info("system audio format: %s Hz (tap %s Hz, device %s Hz)",
                         msg.get("sample_rate"), msg.get("tap_rate"), msg.get("device_rate"))
            elif event == "device_changed" and msg.get("reason"):
                log.warning("system audio: %s", msg["reason"])
            elif event == "started":
                self.last_error = ""
                self._started_event.set()
                self._set_state(SourceState.RUNNING, "process tap active")
            elif event == "heartbeat":
                self._last_heartbeat = time.monotonic()
            elif event == "device_changed":
                log.info("system audio: default output device changed")
            elif event == "error":
                self.last_error = msg.get("message", "helper error")
                self._set_state(SourceState.ERROR, self.last_error)


class AutoSystemSource:
    """Process tap if it works, otherwise BlackHole. Presents a single "system" source."""

    name = "system"

    def __init__(self, buffer: AudioRingBuffer, backend: str = "auto",
                 blackhole_device: str = "BlackHole", on_status: StatusCallback | None = None,
                 tap_start_timeout: float = 5.0) -> None:
        self.buffer = buffer
        self.backend = backend
        self.blackhole_device = blackhole_device
        self.on_status = on_status
        self.tap_start_timeout = tap_start_timeout
        self.active = None
        self.active_backend = ""

    @property
    def state(self) -> SourceState:
        return self.active.state if self.active else SourceState.STOPPED

    def _blackhole(self):
        from .microphone import MicrophoneSource

        name = find_blackhole()
        if name is None:
            raise RuntimeError("BlackHole is not installed.\n" + BLACKHOLE_SETUP)
        return MicrophoneSource(self.buffer, device=name, on_status=self.on_status, name="system")

    def start(self) -> None:
        tap_ok = helper_available() and process_tap_supported()
        if self.backend == "blackhole" or (self.backend == "auto" and not tap_ok):
            if self.backend == "auto":
                log.warning("process tap unavailable (macOS %s, helper %s); using BlackHole",
                            platform.mac_ver()[0], "built" if helper_available() else "missing")
            self.active, self.active_backend = self._blackhole(), "blackhole"
            self.active.start()
            return
        tap = SystemAudioSource(self.buffer, on_status=self.on_status)
        tap.start()
        if self.backend == "tap" or tap.wait_started(self.tap_start_timeout):
            self.active, self.active_backend = tap, "tap"
            return
        # Tap never came up (e.g. creation failed). Fall back if BlackHole is present.
        if find_blackhole() is None:
            log.warning("process tap did not start (%s) and BlackHole is not installed",
                        tap.last_error or "timeout")
            self.active, self.active_backend = tap, "tap"  # keep retrying the tap
            return
        log.warning("process tap failed (%s); falling back to BlackHole", tap.last_error)
        tap.stop()
        self.active, self.active_backend = self._blackhole(), "blackhole"
        self.active.start()

    def stop(self) -> None:
        if self.active:
            self.active.stop()

    def stats(self) -> dict:
        base = self.active.stats() if self.active else {}
        return {"backend": self.active_backend, **base}
