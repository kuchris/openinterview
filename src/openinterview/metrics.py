"""Latency and memory measurement helpers."""

from __future__ import annotations

import resource
import sys
import threading
from collections import deque

import numpy as np


class RollingStat:
    """Thread-safe window of recent samples with mean / percentiles."""

    def __init__(self, window: int = 200) -> None:
        self._values: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()
        self.count = 0

    def add(self, value: float) -> None:
        with self._lock:
            self._values.append(value)
            self.count += 1

    def summary(self, scale: float = 1000.0, digits: int = 1) -> dict:
        with self._lock:
            if not self._values:
                return {"n": 0}
            arr = np.asarray(self._values) * scale
        return {
            "n": self.count,
            "mean": round(float(arr.mean()), digits),
            "p50": round(float(np.percentile(arr, 50)), digits),
            "p95": round(float(np.percentile(arr, 95)), digits),
            "max": round(float(arr.max()), digits),
        }


def peak_memory_mb() -> dict:
    """Peak process RSS plus MLX peak allocation.

    MLX allocates in unified memory through Metal; that is not fully reflected in RSS, so it
    is reported separately.
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024  # bytes on macOS
    out = {"rss_peak_mb": round(rss_mb, 1)}
    try:
        import mlx.core as mx

        out["mlx_peak_mb"] = round(mx.get_peak_memory() / (1024 * 1024), 1)
    except Exception:
        pass
    return out
