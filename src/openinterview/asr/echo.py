"""Acoustic echo handling for "both" mode.

Mic and system audio are separate streams, so there is no *digital* loopback. But when the
call plays through speakers, the mic hears it again, and the interviewer's words would
appear twice: once as "Them" (system) and once as "Me" (mic). Headphones avoid this
entirely. Without them, two layers catch it:

1. EchoGate (audio, before transcription). Compare the mic segment's loudness envelope with
   the system audio's envelope over the same wall-clock span, allowing 0-600 ms for the
   speaker-to-mic path. Echo is a delayed, filtered copy of the system signal, so its
   envelope correlates strongly with the system envelope at some lag, and almost every
   voiced mic frame coincides with system activity. The user talking (alone or over the
   interviewer) adds mic energy where the system is quiet, which breaks the second
   condition, so double-talk is kept.

2. EchoTextFilter (text, after transcription). A mic final whose words closely match a
   system final from the same moment is echo: dropped if the system final came first,
   retracted if the mic final was already emitted.

This is not full acoustic echo cancellation. Mixed double-talk segments keep any echo
inside them; macOS Voice Processing I/O AEC would be the next step.
"""

from __future__ import annotations

import difflib
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..audio.buffer import AudioRingBuffer
from .filters import normalize_text

HOP = 320  # 20 ms at 16 kHz


def envelope(x: np.ndarray, hop: int = HOP) -> np.ndarray:
    n = x.shape[0] // hop
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    frames = x[: n * hop].reshape(n, hop)
    return np.sqrt(np.mean(np.square(frames), axis=1) + 1e-12)


@dataclass
class EchoDecision:
    echo: bool
    correlation: float
    lag_s: float
    unexplained: float      # fraction of voiced mic frames with a quiet system at that lag
    system_active: float    # fraction of the span where the system was playing


@dataclass
class EchoGateConfig:
    max_lag_s: float = 0.6
    min_lag_s: float = -0.1    # allow for timestamp skew between the two clocks' paths
    min_correlation: float = 0.6
    max_unexplained: float = 0.25
    min_system_active: float = 0.3
    system_silence_rms: float = 0.003
    mic_voiced_db: float = 12.0  # mic frame is voiced if this far above its noise floor
    min_frames: int = 15         # need >= 0.3 s of envelope to decide


class EchoGate:
    def __init__(self, config: EchoGateConfig | None = None) -> None:
        self.cfg = config or EchoGateConfig()

    def decide(self, mic: np.ndarray, system: np.ndarray, system_offset_s: float) -> EchoDecision:
        """mic: the mic span. system: system audio starting `system_offset_s` before the mic
        span starts and covering at least the mic span plus max_lag."""
        c = self.cfg
        me = envelope(mic)
        se = envelope(system)
        none = EchoDecision(False, 0.0, 0.0, 1.0, 0.0)
        if me.size < c.min_frames or se.size < me.size:
            return none
        sys_active = float(np.mean(se > c.system_silence_rms))
        if sys_active < c.min_system_active:
            return EchoDecision(False, 0.0, 0.0, 1.0, sys_active)

        mic_db = 20 * np.log10(me + 1e-9)
        sys_db = 20 * np.log10(se + 1e-9)
        voiced = mic_db > np.percentile(mic_db, 10) + c.mic_voiced_db
        if voiced.sum() < c.min_frames // 2:
            return EchoDecision(False, 0.0, 0.0, 1.0, sys_active)

        hop_s = HOP / 16000
        best = (-1.0, 0.0, 1.0)
        for lag in np.arange(c.min_lag_s, c.max_lag_s + 1e-9, hop_s):
            # mic frame i <-> system frame at time (i*hop - lag) relative to the mic start
            off = int(round((system_offset_s - lag) / hop_s))
            if off < 0 or off + me.size > se.size:
                continue
            seg = sys_db[off: off + me.size]
            if seg.std() < 1e-6 or mic_db.std() < 1e-6:
                continue
            corr = float(np.corrcoef(mic_db, seg)[0, 1])
            if corr > best[0]:
                quiet = se[off: off + me.size] <= c.system_silence_rms
                unexplained = float(np.mean(quiet[voiced]))
                best = (corr, float(lag), unexplained)
        corr, lag, unexplained = best
        echo = (corr >= c.min_correlation and unexplained <= c.max_unexplained)
        return EchoDecision(echo, corr, lag, unexplained, sys_active)

    def check_span(self, mic_audio: np.ndarray, mic_start_time: float,
                   system_buffer: AudioRingBuffer) -> EchoDecision:
        """Fetch the time-aligned system audio for a mic span from its ring buffer."""
        c = self.cfg
        sr = system_buffer.sample_rate
        sys_start_time = mic_start_time - c.max_lag_s
        start = system_buffer.index_at(sys_start_time)
        if start is None:
            return EchoDecision(False, 0.0, 0.0, 1.0, 0.0)
        span = mic_audio.shape[0] + int((c.max_lag_s - c.min_lag_s) * sr)
        res = system_buffer.read(max(0, start), max(0, start) + span)
        if res.samples.shape[0] < span:
            return EchoDecision(False, 0.0, 0.0, 1.0, 0.0)
        return self.decide(mic_audio, res.samples, c.max_lag_s)


@dataclass
class _Final:
    source: str
    segment_id: int
    words: list[str]
    start: float
    end: float


class EchoTextFilter:
    """Match mic finals against system finals that overlap them in time."""

    def __init__(self, min_ratio: float = 0.6, slack_s: float = 1.5, min_words: int = 3,
                 history: int = 20) -> None:
        self.min_ratio = min_ratio
        self.slack_s = slack_s
        self.min_words = min_words
        self._recent: deque[_Final] = deque(maxlen=history)

    def _match(self, a: _Final, b: _Final) -> bool:
        if a.end + self.slack_s < b.start or b.end + self.slack_s < a.start:
            return False
        if min(len(a.words), len(b.words)) < self.min_words:
            return False
        ratio = difflib.SequenceMatcher(a=a.words, b=b.words, autojunk=False).ratio()
        return ratio >= self.min_ratio

    def add(self, source: str, segment_id: int, text: str, start: float,
            end: float) -> tuple[bool, list[int]]:
        """Register a final. Returns (drop_this, mic_segment_ids_to_retract)."""
        f = _Final(source, segment_id, normalize_text(text).split(), start, end)
        if source == "mic":
            if any(o.source == "system" and self._match(f, o) for o in self._recent):
                return True, []
            self._recent.append(f)
            return False, []
        retract = [o.segment_id for o in self._recent if o.source == "mic" and self._match(f, o)]
        if retract:
            self._recent = deque((o for o in self._recent
                                  if not (o.source == "mic" and o.segment_id in retract)),
                                 maxlen=self._recent.maxlen)
        self._recent.append(f)
        return False, retract
