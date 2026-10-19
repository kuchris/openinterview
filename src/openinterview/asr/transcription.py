"""Transcription pipeline: source ring buffers -> chunk jobs -> one inference thread.

Phase 1 cuts fixed-length chunks with an energy gate. Phase 2 replaces the chunker with VAD
segmentation and LocalAgreement partials; the threading and queue layout stay the same.

Threads:
  * one chunker per source: waits on its ring buffer, never touches the model
  * one inference thread: owns the ASR engine (MLX requires same-thread use)

Overload policy: jobs go through a DropOldestQueue. If inference falls behind, the oldest
pending chunk is dropped (and counted) so the transcript stays close to live.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..audio.source import AudioSource
from ..metrics import RollingStat
from ..queues import DropOldestQueue
from .engine import ASREngine

log = logging.getLogger(__name__)


@dataclass
class ChunkJob:
    source: str
    audio: np.ndarray
    start_index: int
    end_index: int
    captured_at: float   # monotonic time of the last sample in the chunk
    enqueued_at: float


@dataclass
class TranscriptEvent:
    source: str          # "mic" | "system"
    text: str
    start: float         # seconds since session start
    end: float
    final: bool
    latency_s: float     # end of audio captured -> text available
    inference_s: float


EventCallback = Callable[[TranscriptEvent], None]


class ChunkedTranscriber:
    def __init__(
        self,
        engine_factory: Callable[[], ASREngine],
        sources: list[AudioSource],
        on_event: EventCallback,
        chunk_seconds: float = 4.0,
        silence_rms: float = 0.004,
        queue_size: int = 4,
    ) -> None:
        self.engine_factory = engine_factory
        self.sources = sources
        self.on_event = on_event
        self.chunk_seconds = chunk_seconds
        self.silence_rms = silence_rms
        self.queue: DropOldestQueue[ChunkJob] = DropOldestQueue(queue_size)
        self.ready = threading.Event()  # set once the model is loaded and warmed up
        self.load_error: BaseException | None = None
        self.session_start = time.monotonic()

        self.queue_wait = RollingStat()
        self.inference = RollingStat()
        self.end_to_end = RollingStat()
        self.rtf = RollingStat()
        self.skipped_silent = 0
        self.lost_samples = 0
        self.jobs_done = 0

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self.session_start = time.monotonic()
        self._worker = threading.Thread(target=self._inference_loop, name="asr-inference",
                                        daemon=True)
        self._worker.start()
        for src in self.sources:
            t = threading.Thread(target=self._chunk_loop, args=(src,), name=f"chunker-{src.name}",
                                 daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, drain: bool = True, timeout: float = 30.0) -> None:
        """Flush the last partial chunk of each source, then optionally drain the queue."""
        deadline = time.monotonic() + timeout
        for src in self.sources:
            src.buffer.close()  # chunkers flush what is left and exit
        for t in self._threads:
            t.join(timeout=max(0.1, deadline - time.monotonic()))
        self._threads.clear()
        if drain:
            while len(self.queue) and time.monotonic() < deadline and self.load_error is None:
                time.sleep(0.05)
        self._stop.set()
        self.queue.close()
        self._worker.join(timeout=max(0.1, deadline - time.monotonic()))

    def stats(self) -> dict:
        return {
            "jobs_done": self.jobs_done,
            "dropped_jobs": self.queue.dropped,
            "skipped_silent": self.skipped_silent,
            "lost_samples": self.lost_samples,
            "queue_depth": len(self.queue),
            "queue_wait_ms": self.queue_wait.summary(),
            "inference_ms": self.inference.summary(),
            "end_to_end_ms": self.end_to_end.summary(),
            "rtf": self.rtf.summary(scale=1.0, digits=3),
        }

    # -- threads -----------------------------------------------------------------------

    def _chunk_loop(self, src: AudioSource) -> None:
        buf = src.buffer
        chunk = int(self.chunk_seconds * buf.sample_rate)
        min_flush = buf.sample_rate // 2  # ignore a trailing remainder under 0.5 s
        pos = buf.total_written
        while True:
            target = pos + chunk
            buf.wait_for(target, timeout=0.25)
            if buf.total_written < target:
                if not buf.closed:
                    continue
                target = buf.total_written  # source stopped: flush the remainder once
                if target - pos < min_flush:
                    break
            res = buf.read(pos, target)
            self.lost_samples += res.lost
            pos = target
            audio = res.samples
            if audio.size == 0:
                continue
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if rms < self.silence_rms:
                self.skipped_silent += 1
                continue
            captured = buf.time_at(target) or time.monotonic()
            job = ChunkJob(src.name, audio, res.start_index, target, captured, time.monotonic())
            if self.queue.put(job) is not None:
                log.warning("inference backlog: dropped oldest chunk")
            if buf.closed and buf.total_written <= pos:
                break

    def _inference_loop(self) -> None:
        try:
            engine = self.engine_factory()
            engine.load()
            engine.warmup()
        except BaseException as exc:  # surfaced to the caller via load_error
            self.load_error = exc
            self.ready.set()
            log.exception("failed to load ASR engine")
            return
        self.session_start = time.monotonic()  # timestamps count from when capture can begin
        self.ready.set()
        while True:
            job = self.queue.get(timeout=0.25)
            if job is None:
                if self._stop.is_set() and len(self.queue) == 0:
                    break
                continue
            started = time.monotonic()
            self.queue_wait.add(started - job.enqueued_at)
            result = engine.transcribe(job.audio)
            done = time.monotonic()
            self.jobs_done += 1
            self.inference.add(result.inference_seconds)
            self.rtf.add(result.rtf)
            latency = done - job.captured_at
            self.end_to_end.add(latency)
            if not result.text:
                continue
            sr = 16000
            t0 = (job.captured_at - job.audio.shape[0] / sr) - self.session_start
            self.on_event(TranscriptEvent(
                source=job.source,
                text=result.text,
                start=round(t0, 2),
                end=round(t0 + job.audio.shape[0] / sr, 2),
                final=True,
                latency_s=latency,
                inference_s=result.inference_seconds,
            ))
