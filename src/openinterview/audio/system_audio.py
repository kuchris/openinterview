"""System audio capture through the native `sysaudio` helper (Core Audio Process Tap).

The helper does the capture, downmix and 16 kHz resampling natively and streams framed
float32 PCM on stdout (see helpers/sysaudio/main.swift). This module reads those frames on a
thread and writes them into the source's ring buffer.

Phase 1 scope: helper path only. Phase 3 adds the BlackHole fallback, restart on failure, and
dual-source echo handling.
"""

from __future__ import annotations

import json
import logging
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


def helper_available(path: Path = HELPER_PATH) -> bool:
    return path.is_file() and path.stat().st_mode & 0o111 != 0


def helper_permissions(path: Path = HELPER_PATH) -> dict:
    out = subprocess.run([str(path), "permissions"], capture_output=True, text=True, timeout=10)
    return json.loads(out.stdout)


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
    name = "system"

    def __init__(
        self,
        buffer: AudioRingBuffer,
        helper: Path = HELPER_PATH,
        on_status: StatusCallback | None = None,
    ) -> None:
        self.buffer = buffer
        self.helper = helper
        self.on_status = on_status
        self.state = SourceState.STOPPED
        self.frames = 0
        self.nonzero_frames = 0
        self.last_capture_latency = 0.0
        self.helper_format: dict = {}
        self._proc: subprocess.Popen | None = None
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if not helper_available(self.helper):
            raise RuntimeError(
                f"system audio helper not found at {self.helper}; run scripts/build_helper.sh"
            )
        self._set_state(SourceState.STARTING, "launching helper")
        self._proc = subprocess.Popen(
            [str(self.helper), "capture", "--rate", str(self.buffer.sample_rate)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        for target, name in ((self._read_stdout, "sys-pcm"), (self._read_stderr, "sys-log")):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                proc.stdin.close()  # helper exits when stdin closes
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        for t in self._threads:
            t.join(timeout=2)
        self._threads.clear()
        self._set_state(SourceState.STOPPED, "")

    def stats(self) -> dict:
        return {
            "device": "Core Audio process tap",
            "state": self.state.value,
            "frames": self.frames,
            "nonzero_frames": self.nonzero_frames,
            "capture_latency_ms": round(self.last_capture_latency * 1000, 1),
            **({"format": self.helper_format} if self.helper_format else {}),
        }

    def _set_state(self, state: SourceState, detail: str) -> None:
        self.state = state
        if self.on_status:
            self.on_status(self.name, state, detail)

    def _on_frame(self, samples: np.ndarray, end_time: float) -> None:
        self.frames += 1
        if samples.any():
            self.nonzero_frames += 1
        self.last_capture_latency = max(0.0, time.monotonic() - end_time)
        self.buffer.write(samples, end_time)

    def _read_stdout(self) -> None:
        proc = self._proc
        try:
            read_frames(proc.stdout, self._on_frame)
        except Exception as exc:
            log.error("system audio stream error: %s", exc)
        if self._proc is proc and proc.poll() is not None and proc.returncode != 0:
            self._set_state(SourceState.ERROR, f"helper exited with code {proc.returncode}")

    def _read_stderr(self) -> None:
        proc = self._proc
        for raw in proc.stderr:
            try:
                msg = json.loads(raw)
            except ValueError:
                log.debug("sysaudio: %s", raw.strip())
                continue
            event = msg.get("event")
            if event == "format":
                self.helper_format = msg
            elif event == "started":
                self._set_state(SourceState.RUNNING, "process tap active")
            elif event == "error":
                self._set_state(SourceState.ERROR, msg.get("message", "helper error"))
