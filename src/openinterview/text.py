"""Language-aware tokens for text with and without spaces.

English splits into words on spaces. Chinese and Japanese have no spaces, so each Han /
kana character is its own token; runs of Latin letters or digits inside them ("我用Python寫
API") stay whole words. Everything that compares, counts or re-joins transcript words
(LocalAgreement, repeat and duplicate filters, echo matching, question detection) uses
these, so the same code works for English, Japanese, Mandarin and Cantonese.
"""

from __future__ import annotations

import re

# Han (CJK unified + extension A + compatibility), hiragana, katakana, half-width katakana.
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿ｦ-ﾟ"
# Full-width punctuation: separate tokens, joined without spaces.
_CJK_PUNCT = "　-〿！-／：-＠［-｀｛-･"

_TOKEN = re.compile(rf"[{_CJK}]|[{_CJK_PUNCT}]|[^\s{_CJK}{_CJK_PUNCT}]+")
_HAS_CJK = re.compile(rf"[{_CJK}]")
_NOSPACE = re.compile(rf"[{_CJK}{_CJK_PUNCT}]")


def has_cjk(text: str) -> bool:
    return bool(_HAS_CJK.search(text))


def tokens(text: str) -> list[str]:
    """Words for spaced text; single characters for Chinese / Japanese."""
    return _TOKEN.findall(text)


def _needs_space(left: str, right: str) -> bool:
    return bool(left and right) and not (_NOSPACE.match(left[-1]) or _NOSPACE.match(right[0]))


def join(parts: list[str]) -> str:
    """Inverse of tokens(): spaces only between two non-CJK pieces."""
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out += (" " if _needs_space(out, p) else "") + p
    return out


QUESTION_MARKS = "?？"
TERMINALS = ".?!。？！"
