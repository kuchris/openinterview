"""LocalAgreement-2 for stable partial transcripts.

While a segment is still being spoken, the same growing audio is re-transcribed about once a
second. Each pass may revise its last few words. A word prefix is *committed* only when two
consecutive hypotheses agree on it (compared after normalisation). Committed words never
change or repeat for the rest of the segment, so the UI can show them as stable text, with
the uncommitted tail as a lighter "still changing" suffix.

Reference: Macháček et al., "Turning Whisper into Real-Time Transcription System" (2023).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NORM = re.compile(r"[^\w']+")


def normalize_word(word: str) -> str:
    return _NORM.sub("", word.lower())


def common_prefix_len(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if normalize_word(x) != normalize_word(y):
            break
        n += 1
    return n


@dataclass
class PartialState:
    committed: list[str] = field(default_factory=list)
    previous: list[str] = field(default_factory=list)
    tail: list[str] = field(default_factory=list)

    @property
    def stable(self) -> str:
        return " ".join(self.committed)

    @property
    def unstable(self) -> str:
        return " ".join(self.tail)


class LocalAgreement:
    """One instance per (source, segment)."""

    def __init__(self) -> None:
        self.state = PartialState()

    def update(self, hypothesis: str) -> PartialState:
        words = hypothesis.split()
        s = self.state
        k = len(s.committed)
        # Only words after the committed prefix are candidates. If the new hypothesis
        # rewrote committed words, its alignment is unreliable: keep the commit and do not
        # extend it this round.
        if common_prefix_len(s.committed, words) == k:
            agreed = common_prefix_len(s.previous[k:], words[k:])
            s.committed.extend(words[k:k + agreed])
            k = len(s.committed)
        s.tail = words[k:]  # shown after the committed text; empty if the hypothesis is shorter
        s.previous = words
        return s
