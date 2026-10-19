"""VAD-driven speech segmentation, as a pure state machine.

Input: one speech probability per fixed-size frame, addressed by absolute sample index.
Output: boundary actions telling the pipeline which audio span to transcribe:

  * ``partial``: speech is ongoing; transcribe [start, now) for a live hypothesis
  * ``final``:   the segment is complete; transcribe [start, end) once more for the final text

There is no audio or threading here, which keeps the logic deterministic and testable.

Rules (hysteresis as in Silero's reference get_speech_timestamps):
  * Onset when ``min_speech`` worth of consecutive frames are >= ``threshold``. The segment
    starts ``pre_roll`` before the first speech frame, so soft word onsets are kept.
  * During speech, a frame counts as silence when its prob is < ``neg_threshold``.
    ``min_silence`` of continuous silence closes the segment, which ends ``post_pad``
    after the last voiced frame.
  * Segments never exceed ``max_segment``. At the limit, the cut goes at the least-voiced
    frame in the last ``cut_search`` window (normally a pause between words), and the next
    segment continues from there without a gap or overlap.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegmentAction:
    kind: str            # "partial" | "final"
    segment_id: int
    start: int           # absolute sample index, inclusive
    end: int             # absolute sample index, exclusive
    onset: int           # first voiced sample of the segment (for latency metrics)
    speech_end: int      # end of last voiced frame seen so far
    forced: bool = False  # final caused by max_segment, not by silence


@dataclass
class SegmenterConfig:
    sample_rate: int = 16000
    frame: int = 512
    threshold: float = 0.5
    neg_threshold: float = 0.35
    min_speech_s: float = 0.16
    min_silence_s: float = 0.6
    pre_roll_s: float = 0.2
    post_pad_s: float = 0.1
    max_segment_s: float = 15.0
    cut_search_s: float = 2.0
    partial_interval_s: float = 1.0
    first_partial_s: float = 0.5  # first partial this long after speech onset
    min_partial_s: float = 0.5   # don't run partials on less audio than this


@dataclass
class _Active:
    segment_id: int
    start: int
    onset: int
    speech_end: int
    silence_run: int = 0
    last_partial_end: int = 0
    partials: int = 0
    history: deque = field(default_factory=deque)  # (frame_start, prob) within segment


class Segmenter:
    def __init__(self, config: SegmenterConfig | None = None) -> None:
        self.cfg = c = config or SegmenterConfig()
        sr = c.sample_rate
        self._min_speech_frames = max(1, round(c.min_speech_s * sr / c.frame))
        self._min_silence = int(c.min_silence_s * sr)
        self._pre_roll = int(c.pre_roll_s * sr)
        self._post_pad = int(c.post_pad_s * sr)
        self._max_segment = int(c.max_segment_s * sr)
        self._cut_search = int(c.cut_search_s * sr)
        self._partial_interval = int(c.partial_interval_s * sr)
        self._first_partial = int(c.first_partial_s * sr)
        self._min_partial = int(c.min_partial_s * sr)
        self._next_id = 0
        self._speech_run: list[int] = []  # frame starts of the current candidate onset
        self._active: _Active | None = None
        self._last_end = 0                # end of the previous segment (no overlap)
        self._frame_end = 0

    @property
    def speaking(self) -> bool:
        return self._active is not None

    def push(self, frame_start: int, prob: float) -> list[SegmentAction]:
        c = self.cfg
        frame_end = frame_start + c.frame
        self._frame_end = frame_end
        seg = self._active
        actions: list[SegmentAction] = []

        if seg is None:
            if prob >= c.threshold:
                self._speech_run.append(frame_start)
                if len(self._speech_run) >= self._min_speech_frames:
                    onset = self._speech_run[0]
                    start = max(onset - self._pre_roll, self._last_end, 0)
                    self._active = seg = _Active(self._next_id, start, onset, frame_end,
                                                 last_partial_end=start)
                    self._next_id += 1
                    for fs in self._speech_run:
                        seg.history.append((fs, 1.0))
                    self._speech_run = []
            else:
                self._speech_run = []
            return actions

        seg.history.append((frame_start, prob))
        if prob >= c.neg_threshold:
            seg.speech_end = frame_end
            seg.silence_run = 0
        else:
            seg.silence_run += c.frame

        if seg.silence_run >= self._min_silence:
            end = min(seg.speech_end + self._post_pad, frame_end)
            actions.append(self._final(seg, end, forced=False))
            self._active = None
            return actions

        if frame_end - seg.start >= self._max_segment:
            cut = self._choose_cut(seg, frame_end)
            actions.append(self._final(seg, cut, forced=True))
            nxt = _Active(self._next_id, cut, cut, max(seg.speech_end, cut),
                          last_partial_end=cut)
            self._next_id += 1
            nxt.history.extend((fs, p) for fs, p in seg.history if fs >= cut)
            self._active = seg = nxt

        due = (frame_end - seg.onset >= self._first_partial if seg.partials == 0
               else frame_end - seg.last_partial_end >= self._partial_interval)
        if due and frame_end - seg.start >= self._min_partial:
            seg.last_partial_end = frame_end
            seg.partials += 1
            actions.append(SegmentAction("partial", seg.segment_id, seg.start, frame_end,
                                         seg.onset, seg.speech_end))
        return actions

    def flush(self) -> list[SegmentAction]:
        """Close any open segment (source stopped)."""
        seg = self._active
        if seg is None:
            return []
        self._active = None
        return [self._final(seg, max(seg.speech_end, min(self._frame_end, seg.speech_end
                                                           + self._post_pad)), forced=False)]

    def _final(self, seg: _Active, end: int, forced: bool) -> SegmentAction:
        self._last_end = end
        return SegmentAction("final", seg.segment_id, seg.start, end, seg.onset,
                             min(seg.speech_end, end), forced)

    def _choose_cut(self, seg: _Active, frame_end: int) -> int:
        lo = frame_end - self._cut_search
        candidates = [(p, fs) for fs, p in seg.history if fs >= lo and fs > seg.start]
        if not candidates:
            return frame_end
        # Lowest probability wins; among ties prefer the latest frame (longer first segment).
        _, fs = min(candidates, key=lambda x: (round(x[0], 3), -x[1]))
        return fs
