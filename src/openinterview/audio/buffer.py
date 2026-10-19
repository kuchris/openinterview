"""Bounded, timestamped ring buffer for mono float32 audio.

One writer (the capture callback) and any number of readers. Samples are addressed by an
absolute index (total samples ever written), so readers can track their position without
the buffer keeping per-reader state. Memory is fixed at construction; old audio is
overwritten, and a reader that falls too far behind is told how many samples it lost.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReadResult:
    samples: np.ndarray  # float32, shape (n,); a copy owned by the caller
    start_index: int     # absolute index of samples[0]
    lost: int            # samples requested but already overwritten


class AudioRingBuffer:
    def __init__(self, capacity_seconds: float, sample_rate: int = 16000) -> None:
        if capacity_seconds <= 0:
            raise ValueError("capacity_seconds must be positive")
        self.sample_rate = sample_rate
        self.capacity = int(capacity_seconds * sample_rate)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._total = 0
        # Monotonic time of the sample at _anchor_index; used to timestamp any index.
        self._anchor_index = 0
        self._anchor_time: float | None = None
        self._cond = threading.Condition(threading.Lock())
        self._closed = False

    @property
    def total_written(self) -> int:
        with self._cond:
            return self._total

    @property
    def oldest_index(self) -> int:
        with self._cond:
            return max(0, self._total - self.capacity)

    def write(self, samples: np.ndarray, end_time: float) -> None:
        """Append samples. `end_time` is the monotonic time just after the last sample.

        Called from the audio callback: no allocation beyond the slice views, one short lock.
        """
        n = samples.shape[0]
        if n == 0:
            return
        if n > self.capacity:
            samples = samples[-self.capacity:]
            skipped = n - self.capacity
        else:
            skipped = 0
        m = samples.shape[0]
        with self._cond:
            pos = (self._total + skipped) % self.capacity
            first = min(m, self.capacity - pos)
            self._buf[pos:pos + first] = samples[:first]
            if first < m:
                self._buf[:m - first] = samples[first:]
            self._total += n
            self._anchor_index = self._total
            self._anchor_time = end_time
            self._cond.notify_all()

    def read(self, start: int, end: int | None = None) -> ReadResult:
        """Copy samples in [start, end). `end` defaults to everything written so far."""
        with self._cond:
            total = self._total
            end = total if end is None else min(end, total)
            oldest = max(0, total - self.capacity)
            lost = max(0, oldest - start)
            start = max(start, oldest)
            if end <= start:
                return ReadResult(np.zeros(0, dtype=np.float32), start, lost)
            n = end - start
            out = np.empty(n, dtype=np.float32)
            pos = start % self.capacity
            first = min(n, self.capacity - pos)
            out[:first] = self._buf[pos:pos + first]
            if first < n:
                out[first:] = self._buf[:n - first]
            return ReadResult(out, start, lost)

    def wait_for(self, index: int, timeout: float | None = None) -> bool:
        """Block until at least `index` samples have been written (or closed / timeout)."""
        with self._cond:
            return self._cond.wait_for(lambda: self._total >= index or self._closed, timeout)

    def time_at(self, index: int) -> float | None:
        """Monotonic capture time of the sample at absolute `index` (None before first write)."""
        with self._cond:
            if self._anchor_time is None:
                return None
            return self._anchor_time - (self._anchor_index - index) / self.sample_rate

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    @property
    def closed(self) -> bool:
        return self._closed
