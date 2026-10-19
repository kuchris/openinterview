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
from .echo import EchoGate, EchoTextFilter
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
    """One supervised helper process per audio source.

    If the helper dies mid-session (crash, killed, system service hiccup), it is relaunched
    with backoff. Audio continues from the live position, and any volatile text that was on
    screen is cleared, since that hypothesis died with the process.
    """

    def __init__(self, owner: "AppleStreamingTranscriber", src: AudioSource,
                 command: list[str] | None = None) -> None:
        self.owner = owner
        self.src = src
        self.command = command  # override for tests
        self.proc: subprocess.Popen | None = None
        self.ready = threading.Event()   # first launch finished (ready or error)
        self.error: str | None = None
        self.segment_id = 0
        self.volatile = ""
        self.helper_rss_mb = 0.0
        self.restarts = 0
        self.threads: list[threading.Thread] = []

    def start(self, args: list[str]) -> None:
        self.args = args
        t = threading.Thread(target=self._supervise, name=f"apple-sup-{self.src.name}",
                             daemon=True)
        t.start()
        self.threads.append(t)

    def _supervise(self) -> None:
        backoff = 1.0
        while True:
            cmd = self.command or [str(HELPER_PATH), "stream", *self.args]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, bufsize=0)
            self.proc = proc
            launch_ready = threading.Event()
            reader = threading.Thread(target=self._read_results, args=(proc, launch_ready),
                                      name=f"apple-results-{self.src.name}", daemon=True)
            reader.start()
            launch_ready.wait(120)
            self.ready.set()
            if self.error and self.restarts == 0:
                proc.kill()
                return  # fails on first launch: report, don't loop
            finished = self._pump(proc)
            reader.join(timeout=30)
            if finished or self.src.buffer.closed:
                return
            # helper died while audio was still flowing
            self.restarts += 1
            log.warning("appleasr for %s exited (%s); restarting", self.src.name, proc.poll())
            self.owner._on_helper_restart(self)
            time.sleep(backoff)
            backoff = min(backoff * 2, 15.0)

    def _pump(self, proc: subprocess.Popen) -> bool:
        """Ring buffer -> helper stdin, as audio arrives. True if the source ended normally."""
        buf = self.src.buffer
        pos = buf.total_written
        stdin = proc.stdin
        finished = False
        try:
            while True:
                if proc.poll() is not None:
                    return False
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
            finished = True
        except (BrokenPipeError, ValueError, OSError):
            return False
        finally:
            try:
                stdin.close()  # helper finalizes everything and exits
            except Exception:
                pass
        if finished:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
        return finished

    def _read_results(self, proc: subprocess.Popen, launch_ready: threading.Event) -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except ValueError:
                continue
            kind = msg.get("type")
            if kind == "ready":
                self.error = None
                launch_ready.set()
            elif kind == "error":
                self.error = msg.get("message", "appleasr error")
                launch_ready.set()
            elif kind == "done":
                self.helper_rss_mb = max(self.helper_rss_mb, msg.get("max_rss_mb", 0.0))
            elif kind == "result":
                self.owner._on_result(self, msg)
        launch_ready.set()


class AppleStreamingTranscriber:
    def __init__(self, sources: list[AudioSource], on_event: EventCallback, locale: str = "en-US",
                 vocabulary: str = "", fast: bool = True, merge_gap_s: float = 1.2,
                 echo_suppression: bool = True) -> None:
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
                         "duplicates": 0, "partials": 0, "finals": 0, "echo_gated": 0,
                         "echo_text_dropped": 0, "echo_retracted": 0}
        names = {s.name for s in sources}
        dual = {"mic", "system"} <= names and echo_suppression
        self._system_buffer = next((s.buffer for s in sources if s.name == "system"), None)
        self.echo_gate = EchoGate() if dual else None
        self.echo_text = EchoTextFilter() if dual else None
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
            st.src.buffer.close()  # pumps flush, helpers finalize and exit
        for st in self.streams:
            for t in st.threads:
                t.join(timeout=max(0.1, deadline - time.monotonic()))
            if st.proc and st.proc.poll() is None:
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
            "helper_restarts": sum(st.restarts for st in self.streams),
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
            if self.echo_gate is not None and st.src.name == "mic":
                lo = int(msg["start"] * SR)
                audio = buf.read(lo, int(msg["end"] * SR)).samples
                if self.echo_gate.check_span(audio, start_t, self._system_buffer).echo:
                    self.counters["echo_gated"] += 1
                    self.on_event(TranscriptEvent("final", st.src.name, seg, "",
                                                  self._rel(start_t), self._rel(end_t)))
                    return
            if self.echo_text is not None:
                drop, retract = self.echo_text.add(st.src.name, seg, text, self._rel(start_t),
                                                   self._rel(end_t))
                for seg_id in retract:
                    self.counters["echo_retracted"] += 1
                    self.on_event(TranscriptEvent("retract", "mic", seg_id, "",
                                                  self._rel(start_t), self._rel(end_t)))
                if drop:
                    self.counters["echo_text_dropped"] += 1
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

    def _on_helper_restart(self, st: _Stream) -> None:
        with self._lock:
            if st.volatile:  # clear the stale partial on screen
                st.volatile = ""
                self.on_event(TranscriptEvent("final", st.src.name, st.segment_id, "", 0.0, 0.0))
                st.segment_id += 1

    def _tick_loop(self) -> None:
        while not self._stop.wait(0.1):
            with self._lock:
                speaking = {st.src.name: bool(st.volatile) for st in self.streams}
                for u in self._sentences.tick(time.monotonic(), speaking):
                    self._emit_utterance(u)

    def _emit_utterance(self, u) -> None:
        self.on_event(TranscriptEvent("utterance", u.source, -1, u.text, u.start, u.end))
