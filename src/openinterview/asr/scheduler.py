"""Inference job scheduling and overload policy.

* Finals are never superseded. They wait in a bounded FIFO, and only if more than
  ``max_finals`` pile up is the oldest dropped (counted in ``finals_dropped``).
* Partials are disposable. Each source keeps at most one pending partial; a newer one
  replaces it (``partials_superseded``). A partial is also discarded once its segment's
  final is queued.
* Finals are served before partials. While the final backlog is at least
  ``degrade_at``, new partials are rejected (``partials_skipped``) and the pipeline reports
  itself as degraded. Live captions then pause, but finished sentences keep flowing.

Producers never block.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Job:
    kind: str              # "partial" | "final"
    source: str
    segment_id: int
    audio: Any             # np.ndarray float32
    start: int             # absolute sample indices
    end: int
    start_time: float      # monotonic capture time of audio[0]
    end_time: float        # monotonic capture time of the last sample
    onset_time: float      # monotonic capture time of first voiced sample
    speech_end_time: float  # monotonic capture time of last voiced sample
    speech_ratio: float
    enqueued_at: float
    forced: bool = False


class JobScheduler:
    def __init__(self, max_finals: int = 8, degrade_at: int = 2) -> None:
        self.max_finals = max_finals
        self.degrade_at = degrade_at
        self._finals: deque[Job] = deque()
        self._partials: dict[str, Job] = {}
        self._cond = threading.Condition()
        self._closed = False
        self.finals_dropped = 0
        self.partials_superseded = 0
        self.partials_skipped = 0

    @property
    def degraded(self) -> bool:
        with self._cond:
            return len(self._finals) >= self.degrade_at

    def put(self, job: Job) -> None:
        with self._cond:
            if self._closed:
                return
            if job.kind == "final":
                pending = self._partials.get(job.source)
                if pending is not None and pending.segment_id == job.segment_id:
                    del self._partials[job.source]
                if len(self._finals) >= self.max_finals:
                    self._finals.popleft()
                    self.finals_dropped += 1
                self._finals.append(job)
            else:
                if len(self._finals) >= self.degrade_at:
                    self.partials_skipped += 1
                    return
                if job.source in self._partials:
                    self.partials_superseded += 1
                self._partials[job.source] = job
            self._cond.notify()

    def get(self, timeout: float | None = None) -> Job | None:
        with self._cond:
            ready = lambda: self._finals or self._partials or self._closed  # noqa: E731
            if not self._cond.wait_for(ready, timeout):
                return None
            if self._finals:
                return self._finals.popleft()
            if self._partials:
                # Oldest pending partial across sources first, so no source starves.
                source = min(self._partials, key=lambda s: self._partials[s].enqueued_at)
                return self._partials.pop(source)
            return None

    def pending(self) -> int:
        with self._cond:
            return len(self._finals) + len(self._partials)

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
