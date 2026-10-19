"""Replay an audio file into a ring buffer as if it were live capture.

Used by tests, the benchmark, and `transcribe-file`, so the exact live pipeline can be
exercised without a microphone.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from .buffer import AudioRingBuffer
from .source import SourceState, StatusCallback


def load_audio(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Decode any format ffmpeg understands to mono float32 at `sample_rate`."""
    from mlx_whisper.audio import load_audio as _load

    return np.asarray(_load(path, sr=sample_rate), dtype=np.float32)


class FileSource:
    def __init__(
        self,
        buffer: AudioRingBuffer,
        audio: np.ndarray,
        name: str = "mic",
        speed: float = 1.0,
        block_ms: int = 32,
        tail_silence_s: float = 1.0,
        on_status: StatusCallback | None = None,
    ) -> None:
        self.buffer = buffer
        self.name = name
        self.speed = speed  # 1.0 = real time; 0 = as fast as possible
        self.block = int(buffer.sample_rate * block_ms / 1000)
        pad = np.zeros(int(tail_silence_s * buffer.sample_rate), dtype=np.float32)
        self.audio = np.concatenate([audio.astype(np.float32, copy=False), pad])
        self.on_status = on_status
        self.state = SourceState.STOPPED
        self.done = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop.clear()
        self.state = SourceState.RUNNING
        self._thread = threading.Thread(target=self._run, name=f"file-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.state = SourceState.STOPPED

    def stats(self) -> dict:
        return {"device": "file", "state": self.state.value}

    def _run(self) -> None:
        sr = self.buffer.sample_rate
        t0 = time.monotonic()
        for pos in range(0, self.audio.shape[0], self.block):
            if self._stop.is_set():
                break
            chunk = self.audio[pos:pos + self.block]
            if self.speed > 0:
                due = t0 + (pos + chunk.shape[0]) / sr / self.speed
                delay = due - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            self.buffer.write(chunk, time.monotonic())
        self.state = SourceState.STOPPED
        self.done.set()
