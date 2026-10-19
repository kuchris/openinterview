"""Common interface for audio sources feeding a ring buffer."""

from __future__ import annotations

import enum
from typing import Callable, Protocol

from .buffer import AudioRingBuffer


class SourceState(str, enum.Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    RECONNECTING = "reconnecting"
    ERROR = "error"


StatusCallback = Callable[[str, SourceState, str], None]  # (source name, state, detail)


class AudioSource(Protocol):
    name: str              # "mic" or "system": also the transcript speaker label key
    buffer: AudioRingBuffer
    state: SourceState

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def stats(self) -> dict: ...
