import threading

import numpy as np
import pytest

from openinterview.audio.buffer import AudioRingBuffer


def ramp(start: int, n: int) -> np.ndarray:
    return np.arange(start, start + n, dtype=np.float32)


def make(capacity_samples: int = 100, sr: int = 100) -> AudioRingBuffer:
    return AudioRingBuffer(capacity_samples / sr, sample_rate=sr)


def test_write_and_read_roundtrip():
    buf = make()
    buf.write(ramp(0, 30), end_time=1.0)
    res = buf.read(0)
    assert res.start_index == 0 and res.lost == 0
    np.testing.assert_array_equal(res.samples, ramp(0, 30))


def test_wraparound_preserves_order():
    buf = make(capacity_samples=100)
    for i in range(0, 250, 30):  # writes cross the end of the array several times
        buf.write(ramp(i, 30), end_time=i)
    total = buf.total_written
    assert total == 270
    res = buf.read(total - 100)
    np.testing.assert_array_equal(res.samples, ramp(170, 100))


def test_overrun_reports_lost_samples():
    buf = make(capacity_samples=100)
    buf.write(ramp(0, 250), end_time=1.0)
    res = buf.read(0)
    assert res.lost == 150
    assert res.start_index == 150
    np.testing.assert_array_equal(res.samples, ramp(150, 100))


def test_write_larger_than_capacity_keeps_tail():
    buf = make(capacity_samples=100)
    buf.write(ramp(0, 10), end_time=0.1)
    buf.write(ramp(10, 333), end_time=1.0)
    assert buf.total_written == 343
    np.testing.assert_array_equal(buf.read(243).samples, ramp(243, 100))


def test_read_returns_copy_not_view():
    buf = make()
    buf.write(ramp(0, 10), end_time=1.0)
    out = buf.read(0).samples
    out[:] = -1
    np.testing.assert_array_equal(buf.read(0).samples, ramp(0, 10))


def test_write_does_not_keep_reference_to_input():
    buf = make()
    src = ramp(0, 10)
    buf.write(src, end_time=1.0)
    src[:] = -1
    np.testing.assert_array_equal(buf.read(0).samples, ramp(0, 10))


def test_read_range_and_future_end_clamped():
    buf = make()
    buf.write(ramp(0, 50), end_time=1.0)
    np.testing.assert_array_equal(buf.read(10, 20).samples, ramp(10, 10))
    assert buf.read(40, 500).samples.size == 10
    assert buf.read(60).samples.size == 0


def test_time_at_uses_latest_anchor():
    buf = make(sr=100)
    assert buf.time_at(0) is None
    buf.write(ramp(0, 100), end_time=10.0)
    assert buf.time_at(100) == pytest.approx(10.0)
    assert buf.time_at(50) == pytest.approx(9.5)


def test_memory_is_bounded():
    buf = make(capacity_samples=100)
    for _ in range(1000):
        buf.write(ramp(0, 64), end_time=0.0)
    assert buf._buf.nbytes == 100 * 4


def test_wait_for_wakes_on_write_and_close():
    buf = make()
    got = []
    t = threading.Thread(target=lambda: got.append(buf.wait_for(20, timeout=2)))
    t.start()
    buf.write(ramp(0, 25), end_time=1.0)
    t.join(1)
    assert got == [True]

    t = threading.Thread(target=lambda: got.append(buf.wait_for(1000, timeout=2)))
    t.start()
    buf.close()
    t.join(1)
    assert not t.is_alive()


def test_concurrent_writer_reader_sees_monotonic_data():
    buf = make(capacity_samples=1000, sr=1000)
    n_blocks, block = 500, 32
    errors = []

    def writer():
        for i in range(n_blocks):
            buf.write(ramp(i * block, block), end_time=float(i))
        buf.close()

    def reader():
        pos = 0
        while True:
            buf.wait_for(pos + 1, timeout=0.5)
            res = buf.read(pos)
            if res.samples.size:
                expected = ramp(res.start_index, res.samples.size)
                if not np.array_equal(res.samples, expected):
                    errors.append(res.start_index)
                pos = res.start_index + res.samples.size
            elif buf.closed:
                break

    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert not errors
