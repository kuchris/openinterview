"""Streaming transcription pipeline: VAD segmentation, partials, finals, utterances.

Threads:
  * one segmenter thread per source: reads 32 ms frames from the source's ring buffer, runs
    VAD, and turns segment boundaries into jobs. It never touches the ASR model.
  * one inference thread: owns the ASR engine (MLX needs same-thread use). It also owns
    every piece of transcript state (LocalAgreement, duplicate suppression, sentence
    assembly), so none of that needs locking.

Transcription is chunked, not natively streaming. Each partial re-transcribes the open
segment, which is capped at ``max_segment_s``; LocalAgreement keeps partial text stable.
Overload policy: see scheduler.py.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from ..audio.source import AudioSource
from ..metrics import RollingStat
from .agreement import LocalAgreement
from .engine import ASREngine
from .filters import DuplicateSuppressor, FilterConfig, clean_result
from .scheduler import Job, JobScheduler
from .segmenter import Segmenter, SegmenterConfig
from .sentences import SentenceAssembler

log = logging.getLogger(__name__)


@dataclass
class TranscriptEvent:
    kind: str            # "partial" | "final" | "utterance"
    source: str          # "mic" | "system"
    segment_id: int      # -1 for utterances
    text: str            # full text (partial: stable + unstable)
    start: float         # seconds since session start
    end: float
    stable: str = ""     # partial only: committed by LocalAgreement
    unstable: str = ""   # partial only: may still change
    latency_s: float = 0.0
    inference_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


EventCallback = Callable[[TranscriptEvent], None]


@dataclass
class PipelineConfig:
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    max_finals: int = 8
    degrade_at: int = 2
    vocabulary_prompt: str = ""      # domain terms that bias decoding (e.g. "Kubernetes, gRPC")
    prompt_with_context: bool = True  # also prompt with the tail of the previous final
    merge_gap_s: float = 1.2


class _SourceState:
    def __init__(self, src: AudioSource, cfg: SegmenterConfig) -> None:
        self.src = src
        self.segmenter = Segmenter(cfg)
        # Recent (frame_start, prob) for computing a job's voiced fraction.
        frames = int(cfg.max_segment_s * cfg.sample_rate / cfg.frame) + 64
        self.probs: deque[tuple[int, float]] = deque(maxlen=frames)

    def speech_ratio(self, start: int, end: int, threshold: float) -> float:
        hits = total = 0
        for fs, p in self.probs:
            if start <= fs < end:
                total += 1
                hits += p >= threshold
        return hits / total if total else 0.0


class StreamingTranscriber:
    def __init__(
        self,
        engine_factory: Callable[[], ASREngine],
        sources: list[AudioSource],
        on_event: EventCallback,
        vad_factory: Callable[[], Callable[[np.ndarray], float]],
        config: PipelineConfig | None = None,
    ) -> None:
        self.cfg = config or PipelineConfig()
        self.engine_factory = engine_factory
        self.vad_factory = vad_factory
        self.on_event = on_event
        self.sources = {s.name: _SourceState(s, self.cfg.segmenter) for s in sources}
        self.scheduler = JobScheduler(self.cfg.max_finals, self.cfg.degrade_at)
        self.ready = threading.Event()
        self.load_error: BaseException | None = None
        self.session_start = time.monotonic()
        self.engine: ASREngine | None = None

        self.vad_us = RollingStat(2000)
        self.queue_wait = RollingStat()
        self.inference = {"partial": RollingStat(), "final": RollingStat()}
        self.first_partial = RollingStat()   # speech onset -> first partial on screen
        self.partial_lag = RollingStat()     # partial on screen vs. live audio it covers
        self.final_latency = RollingStat()   # end of speech -> final on screen
        self.segmentation = RollingStat()    # end of speech -> final job enqueued
        self.rtf = RollingStat()
        self.counters = {"lost_samples": 0, "segments": 0, "forced_cuts": 0,
                         "filtered": 0, "duplicates": 0, "partials": 0, "finals": 0}

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._worker: threading.Thread | None = None
        self._agreements: dict[tuple[str, int], LocalAgreement] = {}
        self._first_partial_seen: set[tuple[str, int]] = set()
        self._last_final_text: dict[str, str] = {}
        self._dupes = DuplicateSuppressor()
        self._sentences = SentenceAssembler(self.cfg.merge_gap_s)

    # -- lifecycle ---------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._worker = threading.Thread(target=self._inference_loop, name="asr-inference",
                                        daemon=True)
        self._worker.start()
        for st in self.sources.values():
            t = threading.Thread(target=self._segment_loop, args=(st,),
                                 name=f"segmenter-{st.src.name}", daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self, drain: bool = True, timeout: float = 30.0) -> None:
        """Close open segments, finish queued finals (if drain), then stop inference."""
        deadline = time.monotonic() + timeout
        for st in self.sources.values():
            st.src.buffer.close()
        for t in self._threads:
            t.join(timeout=max(0.1, deadline - time.monotonic()))
        self._threads.clear()
        if drain:
            while (self.scheduler.pending() and time.monotonic() < deadline
                   and self.load_error is None):
                time.sleep(0.02)
        self._stop.set()
        self.scheduler.close()
        if self._worker:
            self._worker.join(timeout=max(0.1, deadline - time.monotonic()))

    @property
    def degraded(self) -> bool:
        return self.scheduler.degraded

    def stats(self) -> dict:
        s = self.scheduler
        return {
            **self.counters,
            "pending_jobs": s.pending(),
            "finals_dropped": s.finals_dropped,
            "partials_superseded": s.partials_superseded,
            "partials_skipped": s.partials_skipped,
            "degraded": s.degraded,
            "vad_frame_us": self.vad_us.summary(scale=1e6, digits=0),
            "queue_wait_ms": self.queue_wait.summary(),
            "inference_partial_ms": self.inference["partial"].summary(),
            "inference_final_ms": self.inference["final"].summary(),
            "first_partial_ms": self.first_partial.summary(),
            "partial_lag_ms": self.partial_lag.summary(),
            "segmentation_ms": self.segmentation.summary(),
            "final_latency_ms": self.final_latency.summary(),
            "rtf": self.rtf.summary(scale=1.0, digits=3),
        }

    # -- segmenter threads -------------------------------------------------------------

    def _segment_loop(self, st: _SourceState) -> None:
        try:
            vad = self.vad_factory()
        except Exception as exc:
            log.exception("VAD init failed for %s", st.src.name)
            self.load_error = exc
            return
        buf = st.src.buffer
        frame = self.cfg.segmenter.frame
        pos = buf.total_written
        while True:
            if not buf.wait_for(pos + frame, timeout=0.25):
                if buf.closed:
                    break
                continue
            if buf.total_written < pos + frame:  # closed while waiting
                break
            res = buf.read(pos, pos + frame)
            if res.lost:
                self.counters["lost_samples"] += res.lost
                pos = res.start_index
                continue
            t0 = time.perf_counter()
            prob = vad(res.samples)
            self.vad_us.add(time.perf_counter() - t0)
            st.probs.append((pos, prob))
            for action in st.segmenter.push(pos, prob):
                self._submit(st, action)
            pos += frame
        for action in st.segmenter.flush():
            self._submit(st, action)

    def _submit(self, st: _SourceState, action) -> None:
        buf = st.src.buffer
        if action.kind == "partial" and self.scheduler.degraded:
            self.scheduler.partials_skipped += 1
            return
        res = buf.read(action.start, action.end)
        if res.lost:
            self.counters["lost_samples"] += res.lost
        if res.samples.size == 0:
            return
        now = time.monotonic()
        speech_end_time = buf.time_at(action.speech_end) or now
        if action.kind == "final":
            self.counters["segments"] += 1
            self.counters["forced_cuts"] += action.forced
            self.segmentation.add(now - speech_end_time)
        self.scheduler.put(Job(
            kind=action.kind,
            source=st.src.name,
            segment_id=action.segment_id,
            audio=res.samples,
            start=res.start_index,
            end=action.end,
            start_time=buf.time_at(res.start_index) or now,
            end_time=buf.time_at(action.end) or now,
            onset_time=buf.time_at(action.onset) or now,
            speech_end_time=speech_end_time,
            speech_ratio=st.speech_ratio(action.start, action.end,
                                         self.cfg.segmenter.threshold),
            enqueued_at=now,
            forced=action.forced,
        ))

    # -- inference thread --------------------------------------------------------------

    def _prompt(self, source: str) -> str | None:
        parts = []
        if self.cfg.vocabulary_prompt:
            parts.append(self.cfg.vocabulary_prompt.strip())
        if self.cfg.prompt_with_context and source in self._last_final_text:
            parts.append(self._last_final_text[source][-200:])
        return " ".join(parts) or None

    def _inference_loop(self) -> None:
        try:
            engine = self.engine_factory()
            engine.load()
            engine.warmup()
            self.engine = engine
        except BaseException as exc:
            self.load_error = exc
            self.ready.set()
            log.exception("failed to load ASR engine")
            return
        self.session_start = time.monotonic()
        self.ready.set()
        while True:
            job = self.scheduler.get(timeout=0.1)
            self._tick_sentences()
            if job is None:
                if self._stop.is_set():
                    break
                continue
            try:
                self._process(engine, job)
            except Exception:
                log.exception("transcription failed for %s job", job.kind)
        for u in self._sentences.flush():
            self._emit_utterance(u)

    def _rel(self, t: float) -> float:
        return round(t - self.session_start, 2)

    def _process(self, engine: ASREngine, job: Job) -> None:
        started = time.monotonic()
        self.queue_wait.add(started - job.enqueued_at)
        result = engine.transcribe(job.audio, prompt=self._prompt(job.source))
        self.inference[job.kind].add(result.inference_seconds)
        self.rtf.add(result.rtf)
        text = clean_result(result, job.speech_ratio, self.cfg.filters)
        done = time.monotonic()
        key = (job.source, job.segment_id)
        start = self._rel(job.start_time)
        end = round(start + (job.end - job.start) / self.cfg.segmenter.sample_rate, 2)

        if job.kind == "partial":
            if not text:
                return
            state = self._agreements.setdefault(key, LocalAgreement()).update(text)
            if key not in self._first_partial_seen:
                self._first_partial_seen.add(key)
                self.first_partial.add(done - job.onset_time)
            self.counters["partials"] += 1
            self.partial_lag.add(done - job.end_time)
            self.on_event(TranscriptEvent(
                "partial", job.source, job.segment_id,
                " ".join(x for x in (state.stable, state.unstable) if x),
                start, end, state.stable, state.unstable,
                done - job.onset_time, result.inference_seconds,
            ))
            return

        # final
        self._agreements.pop(key, None)
        self._first_partial_seen.discard(key)
        if not text:
            self.counters["filtered"] += 1
            # Tell the UI to clear any partial shown for this segment.
            self.on_event(TranscriptEvent("final", job.source, job.segment_id, "", start, end,
                                          latency_s=done - job.speech_end_time,
                                          inference_s=result.inference_seconds))
            return
        if self._dupes.is_duplicate(job.source, text, done):
            self.counters["duplicates"] += 1
            self.on_event(TranscriptEvent("final", job.source, job.segment_id, "", start, end))
            return
        self.counters["finals"] += 1
        latency = done - job.speech_end_time
        if not job.forced:  # forced cuts happen mid-speech; not an end-of-speech latency
            self.final_latency.add(latency)
        self._last_final_text[job.source] = text
        self.on_event(TranscriptEvent("final", job.source, job.segment_id, text, start, end,
                                      latency_s=latency, inference_s=result.inference_seconds))
        for u in self._sentences.add_final(job.source, text, start, end, done):
            self._emit_utterance(u)

    def _tick_sentences(self) -> None:
        speaking = {name: st.segmenter.speaking for name, st in self.sources.items()}
        for u in self._sentences.tick(time.monotonic(), speaking):
            self._emit_utterance(u)

    def _emit_utterance(self, u) -> None:
        self.on_event(TranscriptEvent("utterance", u.source, -1, u.text, u.start, u.end))
