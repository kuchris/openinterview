"""Apple on-device streaming ASR (SpeechAnalyzer, macOS 26+) through the `appleasr` helper.

This is genuine streaming. The recognizer consumes audio continuously and revises a
"volatile" tail until it finalizes it, so the Python VAD segmenter, LocalAgreement and
the scheduler are not used. The model runs in a system service, outside this process, so
our memory cost is just the small helper.

Same surface as StreamingTranscriber (start/stop/ready/stats/events), so the CLI, the
benchmark and the dashboard can use either.
"""

from __future__ import annotations

import json
import logging
import struct
import subprocess
import threading
import time
from pathlib import Path

from ..audio.source import AudioSource
from ..config import PROJECT_ROOT
from ..metrics import RollingStat
from .filters import DuplicateSuppressor
from .sentences import SentenceAssembler
from .transcription import EventCallback, TranscriptEvent

log = logging.getLogger(__name__)

HELPER_PATH = PROJECT_ROOT / "bin" / "appleasr"
AUDIO = 0x31554153
FRAME = 512
SR = 16000


def available(path: Path = HELPER_PATH) -> bool:
    return path.is_file()


class _Stream:
    """One helper process per audio source."""

    def __init__(self, owner: "AppleStreamingTranscriber", src: AudioSource) -> None:
        self.owner = owner
        self.src = src
        self.proc: subprocess.Popen | None = None
        self.ready = threading.Event()
        self.error: str | None = None
        self.segment_id = 0
        self.volatile = ""
        self.helper_rss_mb = 0.0
        self.threads: list[threading.Thread] = []

    def start(self, args: list[str]) -> None:
        self.proc = subprocess.Popen([str(HELPER_PATH), "stream", *args], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        for target, name in ((self._read_results, "results"), (self._pump, "pump")):
            t = threading.Thread(target=target, name=f"apple-{name}-{self.src.name}", daemon=True)
            t.start()
            self.threads.append(t)

    def _pump(self) -> None:
        """Ring buffer -> helper stdin, in 32 ms frames, as audio arrives."""
        buf = self.src.buffer
        if not self.ready.wait(60):
            return
        pos = buf.total_written
        stdin = self.proc.stdin
        try:
            while True:
                if not buf.wait_for(pos + FRAME, timeout=0.25):
                    if buf.closed:
                        break
                    continue
                if buf.total_written < pos + FRAME:
                    break
                res = buf.read(pos)  # everything available, at least one frame
                if res.lost:
                    self.owner.counters["lost_samples"] += res.lost
                n = res.samples.size - res.samples.size % FRAME
                if n == 0:
                    continue
                stdin.write(struct.pack("<IIQ", AUDIO, n, res.start_index)
                            + res.samples[:n].astype("<f4", copy=False).tobytes())
                pos = res.start_index + n
            rest = buf.read(pos)
            if rest.samples.size:
                stdin.write(struct.pack("<IIQ", AUDIO, rest.samples.size, rest.start_index)
                            + rest.samples.astype("<f4", copy=False).tobytes())
        except (BrokenPipeError, ValueError):
            pass
        finally:
            try:
                stdin.close()  # helper finalizes everything and exits
            except Exception:
                pass

    def _read_results(self) -> None:
        for line in self.proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.get("type")
            if kind == "ready":
                self.ready.set()
            elif kind == "error":
                self.error = msg.get("message", "appleasr error")
                self.ready.set()
            elif kind == "done":
                self.helper_rss_mb = msg.get("max_rss_mb", 0.0)
            elif kind == "result":
                self.owner._on_result(self, msg)
        self.ready.set()


class AppleStreamingTranscriber:
    def __init__(self, sources: list[AudioSource], on_event: EventCallback, locale: str = "en-US",
                 vocabulary: str = "", fast: bool = True, merge_gap_s: float = 1.2) -> None:
        self.on_event = on_event
        self.args = ["--locale", locale] + (["--fast"] if fast else [])
        if vocabulary:
            self.args += ["--context", vocabulary.rstrip(".")]
        self.streams = [_Stream(self, s) for s in sources]
        self.ready = threading.Event()
        self.load_error: BaseException | None = None
        self.session_start = time.monotonic()
        self.final_latency = RollingStat()
        self.counters = {"lost_samples": 0, "segments": 0, "forced_cuts": 0, "filtered": 0,
                         "duplicates": 0, "partials": 0, "finals": 0}
        self._lock = threading.Lock()  # results from several helpers share transcript state
        self._dupes = DuplicateSuppressor()
        self._sentences = SentenceAssembler(merge_gap_s)
        self._ticker: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def degraded(self) -> bool:
        return False

    def start(self) -> None:
        if not available():
            self.load_error = RuntimeError("bin/appleasr missing: run scripts/build_helper.sh")
            self.ready.set()
            return
        t0 = time.monotonic()
        for st in self.streams:
            st.start(self.args)
        for st in self.streams:
            st.ready.wait(120)
            if st.error:
                self.load_error = RuntimeError(st.error)
        self.load_seconds = time.monotonic() - t0
        self.session_start = time.monotonic()
        self._ticker = threading.Thread(target=self._tick_loop, name="apple-sentences", daemon=True)
        self._ticker.start()
        self.ready.set()

    def stop(self, drain: bool = True, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        for st in self.streams:
            st.src.buffer.close()
        for st in self.streams:
            for t in st.threads:
                t.join(timeout=max(0.1, deadline - time.monotonic()))
            if st.proc:
                try:
                    st.proc.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    st.proc.kill()
        self._stop.set()
        if self._ticker:
            self._ticker.join(2)
        with self._lock:
            for u in self._sentences.flush():
                self._emit_utterance(u)

    def stats(self) -> dict:
        return {
            **self.counters,
            "pending_jobs": 0, "finals_dropped": 0, "partials_superseded": 0,
            "partials_skipped": 0, "degraded": False,
            "final_latency_ms": self.final_latency.summary(),
            "helper_rss_mb": round(sum(st.helper_rss_mb for st in self.streams), 1),
        }

    # -- results -----------------------------------------------------------------------

    def _rel(self, t: float) -> float:
        return round(t - self.session_start, 2)

    def _on_result(self, st: _Stream, msg: dict) -> None:
        now = time.monotonic()
        buf = st.src.buffer
        start_t = buf.time_at(int(msg["start"] * SR)) or now
        end_t = buf.time_at(int(msg["end"] * SR)) or now
        # Latency is measured from where the last recognized word ends, not from the end of
        # the audio range the recognizer has heard (which can run ahead of its text).
        words_end = msg.get("words_end", -1)
        word_t = (buf.time_at(int(words_end * SR)) or end_t) if words_end >= 0 else end_t
        text = msg["text"].strip()
        with self._lock:
            if not msg["final"]:
                st.volatile = text
                self.counters["partials"] += 1
                # No partial-lag metric here: volatile results carry no per-word timing (their
                # word ranges just span the audio heard so far), so text lag is unmeasurable.
                self.on_event(TranscriptEvent("partial", st.src.name, st.segment_id, text,
                                              self._rel(start_t), self._rel(end_t),
                                              stable="", unstable=text, latency_s=now - word_t))
                return
            seg = st.segment_id
            st.segment_id += 1
            st.volatile = ""
            self.counters["segments"] += 1
            if not text or self._dupes.is_duplicate(st.src.name, text, now):
                self.counters["duplicates" if text else "filtered"] += 1
                self.on_event(TranscriptEvent("final", st.src.name, seg, "",
                                              self._rel(start_t), self._rel(end_t)))
                return
            self.counters["finals"] += 1
            latency = now - word_t
            self.final_latency.add(latency)
            self.on_event(TranscriptEvent("final", st.src.name, seg, text, self._rel(start_t),
                                          self._rel(end_t), latency_s=latency))
            for u in self._sentences.add_final(st.src.name, text, self._rel(start_t),
                                               self._rel(end_t), now):
                self._emit_utterance(u)

    def _tick_loop(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                speaking = {st.src.name: bool(st.volatile) for st in self.streams}
                for u in self._sentences.tick(time.monotonic(), speaking):
                    self._emit_utterance(u)

    def _emit_utterance(self, u) -> None:
        self.on_event(TranscriptEvent("utterance", u.source, -1, u.text, u.start, u.end))
