"""AppleStreamingTranscriber result handling, without the native helper."""

import numpy as np

from openinterview.asr.apple_stream import AppleStreamingTranscriber
from openinterview.audio.buffer import AudioRingBuffer


class FakeSource:
    name = "system"

    def __init__(self):
        self.buffer = AudioRingBuffer(30, 16000)
        self.buffer.write(np.zeros(16000 * 10, np.float32), end_time=100.0)  # t=100 at 10 s


def make():
    events = []
    src = FakeSource()
    t = AppleStreamingTranscriber([src], events.append)
    t.session_start = 90.0
    return t, t.streams[0], events


def result(final, text, start, end, words_end=None):
    return {"type": "result", "final": final, "text": text, "start": start, "end": end,
            "words_end": end if words_end is None else words_end}


def test_volatile_then_final_maps_to_partial_and_final_events():
    t, st, events = make()
    t._on_result(st, result(False, "how would", 2.0, 3.0))
    t._on_result(st, result(False, "how would you design", 2.0, 4.0))
    t._on_result(st, result(True, "How would you design a rate limiter?", 2.0, 5.0))
    kinds = [e.kind for e in events]
    assert kinds[:3] == ["partial", "partial", "final"]
    assert events[1].unstable == "how would you design"
    final = events[2]
    assert final.text == "How would you design a rate limiter?"
    assert (final.start, final.end) == (2.0, 5.0)  # stream seconds -> session-relative
    assert "utterance" in kinds                       # ends with '?', released at once
    assert st.segment_id == 1 and st.volatile == ""


def test_final_latency_uses_last_word_end():
    t, st, _ = make()
    t._on_result(st, result(True, "Okay.", 4.0, 9.0, words_end=8.0))
    # buffer time at 8.0 s is 98.0; latency = now - 98.0 > 2 s since the anchor is t=100
    assert t.final_latency.summary(scale=1.0)["n"] == 1


def test_empty_and_duplicate_finals_suppressed():
    t, st, events = make()
    t._on_result(st, result(True, "", 0.0, 1.0))
    t._on_result(st, result(True, "this is a long repeated sentence", 1.0, 3.0))
    t._on_result(st, result(True, "This is a long repeated sentence.", 3.0, 5.0))
    texts = [e.text for e in events if e.kind == "final"]
    assert texts == ["", "this is a long repeated sentence", ""]
    assert t.counters["filtered"] == 1 and t.counters["duplicates"] == 1
