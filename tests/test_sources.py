"""System-audio supervision and backend selection, with a fake helper (no hardware)."""

import sys
import time
from pathlib import Path

import pytest

from openinterview.audio import portaudio, system_audio
from openinterview.audio.buffer import AudioRingBuffer
from openinterview.audio.source import SourceState
from openinterview.audio.system_audio import AutoSystemSource, SystemAudioSource

FAKE = str(Path(__file__).parent / "fixtures" / "fake_sysaudio.py")


def wait_until(pred, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.05)
    return False


def source(*args):
    return SystemAudioSource(AudioRingBuffer(10, 16000),
                             command=[sys.executable, FAKE, *map(str, args)])


def test_helper_crash_is_restarted_and_audio_continues():
    src = source(20, 1)  # 20 frames then exit(1), every launch
    src.start()
    try:
        assert wait_until(lambda: src.restarts >= 2, timeout=10)
        assert src.buffer.total_written >= 2 * 20 * 160
    finally:
        src.stop()
    assert src.state == SourceState.STOPPED


def test_device_change_restarts_immediately_without_backoff(monkeypatch):
    src = source(5, system_audio.EXIT_DEVICE_CHANGED)
    src.start()
    try:
        assert wait_until(lambda: src.device_changes >= 3, timeout=5)  # no 1-2-4 s backoff
    finally:
        src.stop()


def test_stalled_helper_is_killed_and_restarted(monkeypatch):
    monkeypatch.setattr(system_audio, "STALL_SECONDS", 0.5)
    src = source(5, 0, "stall")
    src.start()
    try:
        assert wait_until(lambda: src.restarts >= 1, timeout=8)
    finally:
        src.stop()


def test_idle_output_with_heartbeat_is_not_restarted(monkeypatch):
    monkeypatch.setattr(system_audio, "STALL_SECONDS", 0.5)
    src = source(5, 0, "idle")  # frames stop (nobody playing) but heartbeats continue
    src.start()
    try:
        time.sleep(2.0)
        assert src.restarts == 0 and src.state == SourceState.RUNNING
    finally:
        src.stop()


def test_idle_gap_is_zero_filled_to_keep_timeline():
    import numpy as np

    src = SystemAudioSource(AudioRingBuffer(10, 16000), command=["true"])
    src._on_frame(np.ones(1600, np.float32), end_time=100.1)
    src._on_frame(np.ones(1600, np.float32), end_time=101.1)  # 0.9 s idle in between
    buf = src.buffer
    assert buf.total_written == 1600 + int(0.9 * 16000) + 1600
    assert abs(buf.time_at(buf.total_written) - 101.1) < 1e-6
    assert abs(src.gap_fill_seconds - 0.9) < 1e-6


def test_auto_uses_blackhole_when_tap_unavailable(monkeypatch):
    monkeypatch.setattr(system_audio, "helper_available", lambda *a: False)
    monkeypatch.setattr(system_audio, "find_blackhole", lambda: "BlackHole 2ch")
    started = []

    class FakeInput:
        def __init__(self, buffer, device=None, on_status=None, name="mic"):
            self.name, self.device = name, device
            self.state = SourceState.RUNNING

        def start(self):
            started.append((self.name, self.device))

        def stop(self):
            pass

        def stats(self):
            return {}

    import openinterview.audio.microphone as mic

    monkeypatch.setattr(mic, "MicrophoneSource", FakeInput)
    src = AutoSystemSource(AudioRingBuffer(1, 16000))
    src.start()
    assert src.active_backend == "blackhole"
    assert started == [("system", "BlackHole 2ch")]


def test_auto_without_any_backend_explains_setup(monkeypatch):
    monkeypatch.setattr(system_audio, "helper_available", lambda *a: False)
    monkeypatch.setattr(system_audio, "find_blackhole", lambda: None)
    with pytest.raises(RuntimeError, match="brew install blackhole-2ch"):
        AutoSystemSource(AudioRingBuffer(1, 16000)).start()


def test_portaudio_reset_is_shared_and_rate_limited(monkeypatch):
    calls = []
    monkeypatch.setattr(portaudio.sd, "_terminate", lambda: calls.append("t"))
    monkeypatch.setattr(portaudio.sd, "_initialize", lambda: calls.append("i"))
    monkeypatch.setattr(portaudio, "_last_reset", 0.0)
    assert portaudio.reset() is True
    assert portaudio.reset() is False  # second source within 2 s reuses the same reset
    assert calls == ["t", "i"]
