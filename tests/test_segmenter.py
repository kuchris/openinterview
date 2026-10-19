from openinterview.asr.segmenter import Segmenter, SegmenterConfig

F = 512
SR = 16000


def feed(seg: Segmenter, probs, start_frame: int = 0):
    actions = []
    for i, p in enumerate(probs):
        actions += seg.push((start_frame + i) * F, p)
    return actions


def frames(seconds: float) -> int:
    return round(seconds * SR / F)


def finals(actions):
    return [a for a in actions if a.kind == "final"]


def partials(actions):
    return [a for a in actions if a.kind == "partial"]


def test_silence_produces_nothing():
    seg = Segmenter()
    assert feed(seg, [0.01] * frames(5)) == []
    assert not seg.speaking


def test_short_click_is_ignored():
    seg = Segmenter()  # min_speech 0.16 s = 5 frames
    acts = feed(seg, [0.0] * 10 + [0.9] * 3 + [0.0] * 40)
    assert acts == []


def test_single_utterance_with_preroll_and_postpad():
    seg = Segmenter()
    speech_start, speech_frames = 30, frames(2.0)
    probs = [0.0] * speech_start + [0.9] * speech_frames + [0.0] * frames(1.0)
    acts = feed(seg, probs)
    (final,) = finals(acts)
    onset = speech_start * F
    assert final.onset == onset
    assert final.start == onset - int(0.2 * SR)                     # pre-roll
    assert final.speech_end == (speech_start + speech_frames) * F
    assert final.end == final.speech_end + int(0.1 * SR)            # post-pad
    assert not final.forced


def test_final_waits_for_min_silence():
    seg = Segmenter()
    probs = [0.9] * frames(1.0) + [0.0] * (frames(0.6) - 1)
    assert finals(feed(seg, probs)) == []
    assert seg.speaking
    assert len(finals(seg.push(len(probs) * F, 0.0))) == 1


def test_short_pause_does_not_split():
    seg = Segmenter()
    probs = [0.9] * frames(1) + [0.1] * frames(0.3) + [0.9] * frames(1) + [0.0] * frames(1)
    assert len(finals(feed(seg, probs))) == 1


def test_hysteresis_keeps_mid_probability_as_speech():
    seg = Segmenter()
    probs = [0.9] * frames(1) + [0.4] * frames(1.0) + [0.0] * frames(1)
    (final,) = finals(feed(seg, probs))
    assert final.speech_end == frames(2.0) * F  # 0.4 >= neg_threshold 0.35


def test_partials_every_interval_with_same_segment_id():
    seg = Segmenter()
    acts = feed(seg, [0.9] * frames(3.5) + [0.0] * frames(1))
    ps = partials(acts)
    # first at 0.5 s after onset, then every 1.0 s: 0.5, 1.5, 2.5, 3.5
    assert len(ps) == 4
    assert abs(ps[0].end - ps[0].onset - 0.5 * SR) <= F
    assert {p.segment_id for p in ps} == {finals(acts)[0].segment_id}
    assert all(p.start == 0 for p in ps)
    assert [p.end for p in ps] == sorted(p.end for p in ps)


def test_max_segment_forces_cut_at_quietest_frame():
    cfg = SegmenterConfig(max_segment_s=5.0, cut_search_s=2.0)
    seg = Segmenter(cfg)
    probs = [0.9] * frames(4.0) + [0.36] + [0.9] * frames(2.0) + [0.0] * frames(1)
    acts = feed(seg, probs)
    fs = finals(acts)
    assert len(fs) == 2
    first, second = fs
    assert first.forced and not second.forced
    assert first.end == frames(4.0) * F       # the 0.36 dip
    assert second.start == first.end          # contiguous: no gap, no overlap
    assert second.segment_id == first.segment_id + 1


def test_continuous_speech_never_exceeds_max_segment():
    cfg = SegmenterConfig(max_segment_s=3.0)
    seg = Segmenter(cfg)
    acts = feed(seg, [0.9] * frames(20))
    for f in finals(acts):
        assert f.end - f.start <= 3.0 * SR
    starts = [f.start for f in finals(acts)][1:]
    ends = [f.end for f in finals(acts)][:-1]
    assert starts == ends


def test_preroll_never_overlaps_previous_segment():
    seg = Segmenter()
    probs = [0.9] * frames(1) + [0.0] * frames(0.65) + [0.9] * frames(1) + [0.0] * frames(1)
    a, b = finals(feed(seg, probs))
    assert b.start >= a.end


def test_flush_closes_open_segment():
    seg = Segmenter()
    feed(seg, [0.9] * frames(1.0))
    (final,) = seg.flush()
    assert final.kind == "final" and final.end > final.start
    assert seg.flush() == []
