"""ASR engine interface shared by Whisper and (later) Parakeet backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Segment:
    text: str
    start: float  # seconds, relative to the start of the transcribed audio
    end: float
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    compression_ratio: float = 0.0


@dataclass
class ASRResult:
    text: str
    segments: list[Segment] = field(default_factory=list)
    audio_seconds: float = 0.0
    inference_seconds: float = 0.0

    @property
    def rtf(self) -> float:
        return self.inference_seconds / self.audio_seconds if self.audio_seconds else 0.0


class ASREngine(Protocol):
    name: str

    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult: ...
