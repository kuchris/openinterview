"""Bounded queue with a drop-oldest overload policy.

Producers never block: when the queue is full the oldest item is evicted and returned, so
the caller can count it. This keeps the newest audio flowing when inference falls behind.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Generic, TypeVar

T = TypeVar("T")


class DropOldestQueue(Generic[T]):
    def __init__(self, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._items: deque[T] = deque()
        self._cond = threading.Condition()
        self._closed = False
        self.dropped = 0

    def put(self, item: T) -> T | None:
        """Enqueue without blocking. Returns the evicted item, if any."""
        with self._cond:
            if self._closed:
                raise RuntimeError("queue is closed")
            evicted = None
            if len(self._items) >= self.maxsize:
                evicted = self._items.popleft()
                self.dropped += 1
            self._items.append(item)
            self._cond.notify()
            return evicted

    def get(self, timeout: float | None = None) -> T | None:
        """Dequeue the oldest item. Returns None on timeout or when closed and empty."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._items or self._closed, timeout):
                return None
            if self._items:
                return self._items.popleft()
            return None

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def __len__(self) -> int:
        with self._cond:
            return len(self._items)
