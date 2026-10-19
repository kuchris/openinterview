"""Hallucination guards and duplicate suppression for ASR output.

Whisper-family models invent text on silence, noise, or music ("Thank you for watching!"),
and can loop ("the the the ..."). VAD gating removes most of these at the source; the
filters here catch what gets through.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..text import has_cjk, join, tokens
from .engine import ASRResult, Segment

# Known silence/noise hallucinations. Only dropped when the decode was also unconfident or
# the audio was mostly non-speech, so a genuinely spoken "Thank you." survives.
BLOCKLIST = {
    "thank you for watching",
    "thanks for watching",
    "thank you so much for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by the amaraorg community",
    "transcription by castingwords",
    "thank you",
    "thanks",
    "bye",
    "you",
    # Whisper's Japanese / Chinese silence hallucinations (subtitle credits, outros)
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "字幕由amaraorg社区提供",
    "字幕由amaraorg社區提供",
    "請不吝點贊訂閱轉發打賞支持明鏡與點點欄目",
    "请不吝点赞订阅转发打赏支持明镜与点点栏目",
    "謝謝觀看",
    "谢谢观看",
    "多謝收睇",
    "謝謝",
    "谢谢",
}

_PUNCT = re.compile(r"[^\w\s']")
_SPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _SPACE.sub(" ", _PUNCT.sub("", text.lower())).strip()


@dataclass
class FilterConfig:
    compression_ratio_max: float = 2.4
    logprob_min: float = -1.0
    no_speech_max: float = 0.6
    blocklist_logprob: float = -0.45   # blocklisted phrases need at least this confidence
    blocklist_speech_ratio: float = 0.6  # ... and at least this VAD speech fraction
    max_repeats: int = 2


def collapse_repeats(text: str, max_repeats: int = 2, max_ngram: int = 6) -> str:
    """Collapse runs of the same 1..max_ngram word sequence repeated > max_repeats times
    (characters for Chinese / Japanese)."""
    words = tokens(text)
    if has_cjk(text):  # one token per character: "哈哈哈" is speech, "的的的的的的" a loop
        max_repeats, max_ngram = max(max_repeats, 3), max_ngram * 2
    if len(words) < max_repeats + 1:
        return text
    changed = True
    while changed:
        changed = False
        for n in range(1, max_ngram + 1):
            i = 0
            out: list[str] = []
            while i < len(words):
                gram = [normalize_text(w) for w in words[i:i + n]]
                reps = 1
                while (i + (reps + 1) * n <= len(words)
                       and [normalize_text(w) for w in words[i + reps * n:i + (reps + 1) * n]] == gram):
                    reps += 1
                if len(gram) == n and reps > max_repeats:
                    out.extend(words[i:i + n])
                    i += reps * n
                    changed = True
                else:
                    out.append(words[i])
                    i += 1
            words = out
    return join(words)


def _segment_ok(seg: Segment, cfg: FilterConfig) -> bool:
    if seg.compression_ratio > cfg.compression_ratio_max:
        return False
    if seg.no_speech_prob > cfg.no_speech_max and seg.avg_logprob < cfg.logprob_min:
        return False
    return True


def clean_result(result: ASRResult, speech_ratio: float = 1.0,
                 cfg: FilterConfig | None = None) -> str:
    """Return the usable text of an ASR result ('' if it is all hallucination)."""
    cfg = cfg or FilterConfig()
    segments = [s for s in result.segments if _segment_ok(s, cfg)] if result.segments else []
    if result.segments:
        text = join([s.text for s in segments])
    else:
        text = result.text.strip()
    if not text:
        return ""
    norm = normalize_text(text)
    if norm in BLOCKLIST:
        logprob = min((s.avg_logprob for s in segments), default=0.0)
        if logprob < cfg.blocklist_logprob or speech_ratio < cfg.blocklist_speech_ratio:
            return ""
    return collapse_repeats(text, cfg.max_repeats)


class DuplicateSuppressor:
    """Drops a final that exactly repeats the previous final from the same source.

    Long identical repeats within seconds come from the decoder re-emitting its prompt
    (the previous sentence), or a hallucination recurring across quiet segments. Short
    repeats ("Okay." ... "Okay.") are normal speech, so only repeats of at least
    ``min_words`` words are suppressed.
    """

    def __init__(self, window_s: float = 10.0, min_words: int = 4) -> None:
        self.window_s = window_s
        self.min_words = min_words
        self._last: dict[str, tuple[str, float]] = {}

    def is_duplicate(self, source: str, text: str, t: float) -> bool:
        norm = normalize_text(text)
        prev = self._last.get(source)
        self._last[source] = (norm, t)
        return bool(prev and norm and prev[0] == norm and t - prev[1] <= self.window_s
                    and len(tokens(norm)) >= self.min_words)
