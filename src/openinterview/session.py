"""A transcription session that a UI can start, stop and observe.

Owns the audio sources and the transcriber, keeps a bounded in-memory transcript, and
publishes everything through a single `sink(message: dict)` callback. Messages:

  {"type": "status",     "state": idle|loading|running|stopping|error, ...}
  {"type": "source",     "source": "mic"|"system", "state": ..., "detail": ...}
  {"type": "transcript", "event": TranscriptEvent as dict}
  {"type": "metrics",    ...pipeline / source / memory stats}

The sink is called from worker threads; the web server marshals onto its event loop.
Audio is never stored. Transcript persistence (JSONL) is opt-in per session.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import PROJECT_ROOT, Config

log = logging.getLogger(__name__)

Sink = Callable[[dict], None]
SPEAKER = {"mic": "Me", "system": "Them"}


class Session:
    def __init__(self, cfg: Config, sink: Sink,
                 sources_factory: Callable[[Config, str, Callable], list] | None = None,
                 transcriber_factory: Callable | None = None,
                 max_lines: int = 5000) -> None:
        self.base_cfg = cfg
        self.sink = sink
        self._sources_factory = sources_factory or _default_sources
        self._transcriber_factory = transcriber_factory or _default_transcriber
        self.state = "idle"
        self.error = ""
        self.cfg: Config = cfg
        self.sources: list = []
        self.transcriber = None
        self.lines: deque[dict] = deque(maxlen=max_lines)   # finals, in order
        self.partials: dict[str, dict] = {}                 # source -> latest partial
        self.started_at: float | None = None
        self.listeners: list[Callable[[dict], None]] = []   # e.g. the practice assistant
        self._lock = threading.RLock()
        self._file = None
        self.transcript_path: Path | None = None
        self._metrics_thread: threading.Thread | None = None
        self._stop_metrics = threading.Event()

    # -- control -------------------------------------------------------------------------

    def start(self, source: str | None = None, model: str | None = None,
              mic_device: str | int | None = None, save_transcript: bool | None = None) -> None:
        with self._lock:
            if self.state in ("loading", "running", "stopping"):
                raise RuntimeError(f"session is {self.state}")
            cfg = self.base_cfg
            audio = replace(cfg.audio, source=source or cfg.audio.source,
                            mic_device=mic_device if mic_device not in ("", None)
                            else cfg.audio.mic_device)
            asr = replace(cfg.asr, model=model or cfg.asr.model)
            storage = replace(cfg.storage, save_transcripts=cfg.storage.save_transcripts
                              if save_transcript is None else save_transcript)
            self.cfg = replace(cfg, audio=audio, asr=asr, storage=storage)
            self.lines.clear()
            self.partials.clear()
            self.error = ""
            self._set_state("loading")
        threading.Thread(target=self._start_worker, name="session-start", daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            if self.state not in ("loading", "running"):
                return
            self._set_state("stopping")
        threading.Thread(target=self._stop_worker, name="session-stop", daemon=True).start()

    def wait_state(self, states: set[str], timeout: float) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.state in states:
                return True
            time.sleep(0.05)
        return self.state in states

    # -- workers ---------------------------------------------------------------------------

    def _start_worker(self) -> None:
        cfg = self.cfg
        try:
            self.sources = self._sources_factory(cfg, cfg.audio.source, self._on_source_status)
            self.transcriber = self._transcriber_factory(cfg, self.sources, self._on_event)
            self.transcriber.start()
            self.transcriber.ready.wait()
            if self.transcriber.load_error:
                raise RuntimeError(f"model load failed: {self.transcriber.load_error}")
            if cfg.storage.save_transcripts:
                self._open_transcript_file()
            for src in self.sources:
                src.start()
            with self._lock:
                if self.state == "stopping":
                    return
                self.started_at = time.time()
                self._set_state("running")
            self._stop_metrics.clear()
            self._metrics_thread = threading.Thread(target=self._metrics_loop,
                                                    name="session-metrics", daemon=True)
            self._metrics_thread.start()
        except Exception as exc:
            log.exception("session start failed")
            self._teardown()
            with self._lock:
                self.error = str(exc)
                self._set_state("error")

    def _stop_worker(self) -> None:
        self._teardown()
        with self._lock:
            self._set_state("idle")

    def _teardown(self) -> None:
        self._stop_metrics.set()
        for src in self.sources:
            try:
                src.stop()
            except Exception:
                log.exception("source stop failed")
        if self.transcriber is not None:
            try:
                self.transcriber.stop(drain=True, timeout=20)
            except Exception:
                log.exception("transcriber stop failed")
        if self._metrics_thread:
            self._metrics_thread.join(2)
        self.publish({"type": "metrics", **self.metrics()})
        self.partials.clear()
        if self._file:
            self._file.close()
            self._file = None

    # -- events ------------------------------------------------------------------------------

    def publish(self, msg: dict) -> None:
        try:
            self.sink(msg)
        except Exception:
            log.exception("sink failed")
        for fn in list(self.listeners):
            try:
                fn(msg)
            except Exception:
                log.exception("listener failed")

    def _set_state(self, state: str) -> None:
        self.state = state
        self.publish({"type": "status", **self.status()})

    def _on_source_status(self, name: str, state, detail: str) -> None:
        self.publish({"type": "source", "source": name, "state": state.value, "detail": detail})

    def _on_event(self, ev) -> None:
        d = ev.to_dict()
        d["speaker"] = SPEAKER.get(ev.source, ev.source)
        with self._lock:
            if ev.kind == "partial":
                self.partials[ev.source] = d
            elif ev.kind == "final":
                self.partials.pop(ev.source, None)
                if ev.text:
                    self.lines.append(d)
            elif ev.kind == "retract":
                self.lines = deque((x for x in self.lines
                                    if not (x["source"] == ev.source
                                            and x["segment_id"] == ev.segment_id)),
                                   maxlen=self.lines.maxlen)
            if self._file and (ev.kind in ("final", "utterance", "retract")
                               and (ev.text or ev.kind == "retract")):
                self._file.write(json.dumps(d) + "\n")
                self._file.flush()
        self.publish({"type": "transcript", "event": d})

    def _open_transcript_file(self) -> None:
        folder = PROJECT_ROOT / self.cfg.storage.transcript_dir
        folder.mkdir(parents=True, exist_ok=True)
        self.transcript_path = folder / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
        self._file = self.transcript_path.open("a")

    def _metrics_loop(self) -> None:
        while not self._stop_metrics.wait(1.0):
            self.publish({"type": "metrics", **self.metrics()})

    # -- snapshots ---------------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "state": self.state,
            "error": self.error,
            "source": self.cfg.audio.source,
            "model": self.cfg.asr.model,
            "saving": bool(self._file),
            "transcript_path": str(self.transcript_path) if self._file else None,
            "started_at": self.started_at,
        }

    def metrics(self) -> dict:
        from .metrics import peak_memory_mb

        t = self.transcriber
        out = {"pipeline": t.stats() if t is not None else {},
               "sources": {s.name: s.stats() for s in self.sources},
               "memory": peak_memory_mb()}
        out["degraded"] = bool(getattr(t, "degraded", False)) if t is not None else False
        return out

    def snapshot(self) -> dict:
        with self._lock:
            return {"status": self.status(), "lines": list(self.lines),
                    "partials": dict(self.partials)}

    def export_markdown(self) -> str:
        with self._lock:
            lines = list(self.lines)
        out = [f"# Transcript {datetime.now():%Y-%m-%d %H:%M}", ""]
        for x in lines:
            m, s = divmod(int(x["start"]), 60)
            out.append(f"**{x['speaker']}** [{m:02d}:{s:02d}] {x['text']}  ")
        return "\n".join(out) + "\n"


def _default_sources(cfg: Config, source: str, on_status) -> list:
    from .audio.buffer import AudioRingBuffer
    from .audio.microphone import MicrophoneSource
    from .audio.system_audio import AutoSystemSource

    sr, secs = cfg.audio.sample_rate, cfg.audio.ring_buffer_seconds
    sources = []
    if source in ("mic", "both"):
        sources.append(MicrophoneSource(AudioRingBuffer(secs, sr), cfg.audio.mic_device,
                                        on_status=on_status))
    if source in ("system", "both"):
        sources.append(AutoSystemSource(AudioRingBuffer(secs, sr), cfg.audio.system_backend,
                                        cfg.audio.blackhole_device, on_status=on_status))
    return sources


def _default_transcriber(cfg: Config, sources: list, on_event):
    from .asr.factory import build_transcriber

    return build_transcriber(cfg, sources, on_event)
