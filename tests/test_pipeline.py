"""Pipeline tests with a fake engine and energy VAD (no model download, no audio hardware)."""

import io
import struct
import threading
import time

import numpy as np

from openinterview.asr.engine import ASRResult, Segment
from openinterview.asr.scheduler import Job, JobScheduler
from openinterview.asr.segmenter import SegmenterConfig
from openinterview.asr.transcription import PipelineConfig, StreamingTranscriber
from openinterview.asr.vad import EnergyVAD
from openinterview.audio.buffer import AudioRingBuffer
from openinterview.audio.file_source import FileSource
from openinterview.audio.system_audio import FRAME_MAGIC, read_frames

SR = 16000


class FakeEngine:
    """Returns one word per 0.25 s of audio, so longer partials extend the hypothesis."""

    name = "fake"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = []
        self.thread = None

    def load(self):
        self.thread = threading.get_ident()

    def warmup(self):
        pass

    def transcribe(self, audio, prompt=None):
        assert threading.get_ident() == self.thread, "engine used off its owning thread"
        self.calls.append(audio.size)
        time.sleep(self.delay)
        n = max(1, int(audio.size / SR / 0.25))
        text = " ".join(f"w{i}" for i in range(n)) + "."
        return ASRResult(text, [Segment(text, 0, audio.size / SR, avg_logprob=-0.1)],
                         audio_seconds=audio.size / SR, inference_seconds=self.delay)


def tone(seconds: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SR), np.float32)


def run(audio, engine, speed=0.0, **cfg_kw):
    events = []
    src = FileSource(AudioRingBuffer(30, SR), audio, speed=speed, tail_silence_s=0)
    cfg = PipelineConfig(segmenter=SegmenterConfig(**cfg_kw), prompt_with_context=False)
    pipe = StreamingTranscriber(lambda: engine, [src], events.append,
                                vad_factory=lambda: EnergyVAD(0.01), config=cfg)
    pipe.start()
    assert pipe.ready.wait(5)
    src.start()
    src.done.wait(20)
    pipe.stop(drain=True)
    return events, pipe


def kinds(events, kind):
    return [e for e in events if e.kind == kind]


def test_two_utterances_give_two_finals():
    audio = np.concatenate([silence(0.5), tone(1.5), silence(1.0), tone(1.0), silence(1.0)])
    events, pipe = run(audio, FakeEngine())
    finals = [e for e in kinds(events, "final") if e.text]
    assert len(finals) == 2
    assert finals[0].segment_id != finals[1].segment_id
    assert pipe.counters["segments"] == 2


def test_silence_only_never_calls_engine():
    engine = FakeEngine()
    events, _ = run(silence(3.0), engine)
    assert engine.calls == []
    assert events == []


def test_partials_precede_final_and_grow_stably():
    audio = np.concatenate([tone(3.5), silence(1.0)])
    events, _ = run(audio, FakeEngine(), speed=4.0)
    partials = kinds(events, "partial")
    finals = [e for e in kinds(events, "final") if e.text]
    assert len(finals) == 1 and partials
    assert events.index(partials[-1]) < events.index(finals[0])
    stable = [p.stable for p in partials]
    for a, b in zip(stable, stable[1:]):
        assert b.startswith(a)


def test_long_speech_is_cut_into_bounded_segments():
    audio = np.concatenate([tone(9.0), silence(1.0)])
    engine = FakeEngine()
    events, pipe = run(audio, engine, max_segment_s=4.0)
    assert pipe.counters["forced_cuts"] >= 2
    assert max(engine.calls) <= 4.0 * SR + 512


def test_utterance_emitted_for_final_with_punctuation():
    audio = np.concatenate([tone(1.0), silence(1.0)])
    events, _ = run(audio, FakeEngine())
    (u,) = kinds(events, "utterance")
    assert u.text.endswith(".")


def test_overloaded_engine_keeps_finals_and_skips_partials():
    # Engine 4x slower than real time on partials; finals must still all arrive.
    audio = np.concatenate([np.concatenate([tone(2.0), silence(0.8)]) for _ in range(4)])
    engine = FakeEngine(delay=0.5)
    events, pipe = run(audio, engine, speed=2.0, partial_interval_s=0.25)
    stats = pipe.stats()
    # The fake engine returns identical 8-word text for equal-length segments, so repeats
    # are (correctly) caught by duplicate suppression; none may be dropped by overload.
    assert stats["finals"] + stats["duplicates"] == 4
    assert stats["finals_dropped"] == 0
    assert stats["partials_superseded"] + stats["partials_skipped"] > 0


# -- scheduler ------------------------------------------------------------------------

def job(kind, source="mic", seg=0, t=0.0):
    return Job(kind, source, seg, np.zeros(1, np.float32), 0, 1, 0, 0, 0, 0, 1.0, t)


def test_scheduler_serves_finals_first_and_supersedes_partials():
    s = JobScheduler(max_finals=8, degrade_at=5)
    s.put(job("partial", t=1))
    s.put(job("partial", t=2))
    s.put(job("final", seg=9, t=3))
    assert s.partials_superseded == 1
    assert s.get(0).kind == "final"
    p = s.get(0)
    assert p.kind == "partial" and p.enqueued_at == 2


def test_scheduler_final_cancels_its_segment_partial():
    s = JobScheduler()
    s.put(job("partial", seg=3))
    s.put(job("final", seg=3))
    assert s.get(0).kind == "final"
    assert s.get(0) is None


def test_scheduler_bounds_finals_and_degrades():
    s = JobScheduler(max_finals=3, degrade_at=2)
    for i in range(5):
        s.put(job("final", seg=i))
    assert s.finals_dropped == 2 and s.degraded
    s.put(job("partial", seg=99))
    assert s.partials_skipped == 1
    assert [s.get(0).segment_id for _ in range(3)] == [2, 3, 4]


# -- helper protocol ------------------------------------------------------------------

def test_read_frames_parses_helper_protocol():
    payload = np.arange(5, dtype="<f4")
    frame = struct.pack("<IIQ", FRAME_MAGIC, 5, 2_500_000_000) + payload.tobytes()
    got = []
    read_frames(io.BytesIO(frame * 2), lambda s, t: got.append((s.copy(), t)))
    assert len(got) == 2
    np.testing.assert_array_equal(got[0][0], payload)
    assert got[0][1] == 2.5
