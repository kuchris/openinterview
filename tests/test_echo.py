import numpy as np

from openinterview.asr.echo import EchoGate, EchoTextFilter
from openinterview.audio.buffer import AudioRingBuffer

SR = 16000
rng = np.random.default_rng(7)


def speechlike(seconds: float, seed: int) -> np.ndarray:
    """Syllable-like bursts of shaped noise with pauses (an envelope like real speech)."""
    r = np.random.default_rng(seed)
    out = np.zeros(int(seconds * SR), np.float32)
    t = 0
    while t < out.size:
        n = int(r.uniform(0.08, 0.3) * SR)
        burst = r.standard_normal(n).astype(np.float32) * np.hanning(n).astype(np.float32)
        out[t:t + n] = 0.2 * burst[: max(0, min(n, out.size - t))]
        t += n + int(r.uniform(0.05, 0.25) * SR)
    return out


def room(x: np.ndarray, delay_s: float, gain: float = 0.3) -> np.ndarray:
    """Speaker -> room -> mic: delay, attenuation, smearing, mic noise."""
    d = int(delay_s * SR)
    y = np.concatenate([np.zeros(d, np.float32), x])[: x.size]
    y = np.convolve(y, np.ones(24) / 24, mode="same").astype(np.float32) * gain
    return y + 0.002 * rng.standard_normal(x.size).astype(np.float32)


def check(mic, system, lead=0.6):
    # system audio starts `lead` s before the mic span
    sys_span = np.concatenate([np.zeros(int(lead * SR), np.float32), system,
                               np.zeros(int(0.2 * SR), np.float32)])
    return EchoGate().decide(mic, sys_span, lead)


def test_speaker_echo_is_detected():
    system = speechlike(3.0, 1)
    d = check(room(system, 0.15), system)
    assert d.echo, d
    assert abs(d.lag_s - 0.15) < 0.05


def test_independent_speech_is_not_echo():
    d = check(speechlike(3.0, 2) + 0.002, speechlike(3.0, 1))
    assert not d.echo, d


def test_double_talk_is_kept():
    system = speechlike(3.0, 1)
    user = speechlike(3.0, 3)
    d = check(room(system, 0.15) + user, system)
    assert not d.echo, d


def test_quiet_system_means_no_echo():
    d = check(speechlike(3.0, 2), np.zeros(int(3.0 * SR), np.float32))
    assert not d.echo and d.system_active == 0.0


def test_check_span_aligns_by_capture_time():
    system = speechlike(4.0, 1)
    sys_buf = AudioRingBuffer(10, SR)
    sys_buf.write(system, end_time=104.0)  # system audio spans t=100..104
    mic = room(system, 0.15)[SR:3 * SR]   # mic span t=101..103
    d = EchoGate().check_span(mic, 101.0, sys_buf)
    assert d.echo, d


def test_index_at_inverts_time_at():
    buf = AudioRingBuffer(10, SR)
    buf.write(np.zeros(SR * 2, np.float32), end_time=50.0)
    assert buf.index_at(49.0) == SR
    assert buf.time_at(buf.index_at(49.5)) == 49.5


# -- text layer -------------------------------------------------------------------------

def test_mic_final_matching_earlier_system_final_dropped():
    f = EchoTextFilter()
    assert f.add("system", 1, "How would you design a rate limiter?", 10.0, 12.5) == (False, [])
    drop, _ = f.add("mic", 4, "how would you design a rate limit", 10.2, 12.8)
    assert drop


def test_system_final_retracts_earlier_mic_echo():
    f = EchoTextFilter()
    assert f.add("mic", 4, "how would you design a rate limiter", 10.2, 12.8) == (False, [])
    drop, retract = f.add("system", 1, "How would you design a rate limiter?", 10.0, 12.5)
    assert not drop and retract == [4]


def test_same_words_at_different_times_kept():
    f = EchoTextFilter()
    f.add("system", 1, "Can you explain the difference between a process and a thread", 10, 13)
    drop, _ = f.add("mic", 2, "the difference between a process and a thread is memory", 30, 33)
    assert not drop


def test_short_answers_never_matched():
    f = EchoTextFilter()
    f.add("system", 1, "Okay.", 10, 10.5)
    assert f.add("mic", 2, "Okay.", 10.1, 10.6) == (False, [])
