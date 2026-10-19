"""Pipeline tests with a fake engine (no model download, no audio hardware)."""

import io
import struct
import threading
import time

import numpy as np

from openinterview.asr.engine import ASRResult
from openinterview.asr.transcription import ChunkedTranscriber
from openinterview.audio.buffer import AudioRingBuffer
from openinterview.audio.file_source import FileSource
from openinterview.audio.system_audio import FRAME_MAGIC, read_frames

SR = 16000


class FakeEngine:
    name = "fake"

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.thread = None

    def load(self):
        self.thread = threading.get_ident()

    def warmup(self):
        pass

    def transcribe(self, audio, prompt=None):
        assert threading.get_ident() == self.thread, "engine used off its owning thread"
        self.calls += 1
        time.sleep(self.delay)
        return ASRResult(text=f"chunk{self.calls}", audio_seconds=audio.size / SR,
                         inference_seconds=self.delay)


def tone(seconds: float, amp: float = 0.1) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def run(audio, engine, chunk_seconds=1.0, queue_size=4, speed=0.0):
    events = []
    src = FileSource(AudioRingBuffer(30, SR), audio, speed=speed, tail_silence_s=0)
    pipe = ChunkedTranscriber(lambda: engine, [src], events.append,
                              chunk_seconds=chunk_seconds, queue_size=queue_size)
    pipe.start()
    pipe.ready.wait(5)
    src.start()
    src.done.wait(10)
    pipe.stop(drain=True)
    return events, pipe


def test_chunks_and_flushes_remainder():
    engine = FakeEngine()
    events, pipe = run(tone(3.6), engine)
    assert engine.calls == 4  # 3 full chunks + 0.6 s remainder
    assert [e.source for e in events] == ["mic"] * 4
    assert pipe.stats()["dropped_jobs"] == 0


def test_silence_is_skipped():
    engine = FakeEngine()
    audio = np.concatenate([np.zeros(2 * SR, np.float32), tone(1.0)])
    events, pipe = run(audio, engine)
    assert engine.calls == 1
    assert pipe.skipped_silent == 2


def test_overload_drops_oldest_and_stays_bounded():
    engine = FakeEngine(delay=0.3)  # far slower than the 0.1 s chunks arriving
    events, pipe = run(tone(4.0), engine, chunk_seconds=0.1, queue_size=2, speed=4.0)
    stats = pipe.stats()
    assert stats["dropped_jobs"] > 0
    assert engine.calls + stats["dropped_jobs"] == 40


def test_read_frames_parses_helper_protocol():
    payload = np.arange(5, dtype="<f4")
    frame = struct.pack("<IIQ", FRAME_MAGIC, 5, 2_500_000_000) + payload.tobytes()
    got = []
    read_frames(io.BytesIO(frame * 2), lambda s, t: got.append((s.copy(), t)))
    assert len(got) == 2
    np.testing.assert_array_equal(got[0][0], payload)
    assert got[0][1] == 2.5
