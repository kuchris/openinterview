"""Merge segment finals into complete utterances.

A pause long enough to close a VAD segment (0.6 s) often falls mid-sentence ("so the way I
would approach this is ... um ... with a token bucket"). Downstream consumers, such as the
Phase 5 question detector, want whole utterances. Finals ending without terminal
punctuation are held, and the next final from the same source is appended. An utterance is
released when it ends with . ? or !, when the speaker has been quiet for ``merge_gap_s``,
or when it reaches ``max_hold_s``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TERMINAL = re.compile(r"[.?!][\"')\]]*$")


@dataclass
class Utterance:
    source: str
    text: str
    start: float
    end: float


@dataclass
class _Pending:
    parts: list[str]
    start: float
    end: float
    wall_end: float  # monotonic time the last part arrived


class SentenceAssembler:
    def __init__(self, merge_gap_s: float = 1.2, max_hold_s: float = 30.0) -> None:
        self.merge_gap_s = merge_gap_s
        self.max_hold_s = max_hold_s
        self._pending: dict[str, _Pending] = {}

    def add_final(self, source: str, text: str, start: float, end: float,
                  now: float) -> list[Utterance]:
        text = text.strip()
        if not text:
            return []
        p = self._pending.get(source)
        if p is None:
            p = self._pending[source] = _Pending([text], start, end, now)
        else:
            p.parts.append(text)
            p.end, p.wall_end = end, now
        if _TERMINAL.search(text) or p.end - p.start >= self.max_hold_s:
            return [self._release(source)]
        return []

    def tick(self, now: float, speaking: dict[str, bool]) -> list[Utterance]:
        out = []
        for source in list(self._pending):
            p = self._pending[source]
            if not speaking.get(source, False) and now - p.wall_end >= self.merge_gap_s:
                out.append(self._release(source))
        return out

    def flush(self) -> list[Utterance]:
        return [self._release(s) for s in list(self._pending)]

    def _release(self, source: str) -> Utterance:
        p = self._pending.pop(source)
        return Utterance(source, " ".join(p.parts), p.start, p.end)
