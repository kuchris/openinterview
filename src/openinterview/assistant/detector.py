"""Decide which complete utterances are technical interview questions.

Input is *utterances*: finals already merged into complete sentences (asr/sentences.py).
Partials and fragments are never considered, so incomplete transcripts are not sent.

Heuristics (no model call):
  * at least ``min_words`` words
  * a question form: ends with "?", starts with an interrogative ("how would you ..."),
    or an interview prompt ("explain ...", "walk me through ...", "design ...")
  * optionally (``require_technical``) a technical cue, so small talk such as
    "How are you doing today?" is ignored
Duplicate suppression: a question too similar to a recent one (difflib ratio >= 0.85 on
normalised words) is not asked again.
"""

from __future__ import annotations

import difflib
import re
from collections import deque
from dataclasses import dataclass

from ..asr.filters import normalize_text

INTERROGATIVE = re.compile(
    r"^(so,?\s+|and\s+|okay,?\s+|ok,?\s+|alright,?\s+|now,?\s+|next,?\s+)*"
    r"(what|why|how|when|where|which|who|whose|can you|could you|would you|will you|"
    r"do you|did you|have you|are you|is there|is it|are there|should|tell me|"
    r"explain|describe|walk me through|talk me through|compare|implement|write|design|"
    r"define|give me|show me|what's|how's|whats)\b"
)

TECH_TERMS = {
    "algorithm", "api", "array", "async", "asyncio", "big", "binary", "bug", "cache",
    "class", "complexity", "concurrency", "consistency", "container", "data", "database",
    "debug", "deploy", "design", "distributed", "docker", "dynamic", "error", "function",
    "gil", "graph", "hash", "heap", "http", "index", "inheritance", "interface", "kubernetes",
    "latency", "linked", "list", "lock", "memory", "microservice", "multiprocessing",
    "network", "object", "optimize", "performance", "pointer", "process", "python", "query",
    "queue", "recursion", "redis", "rest", "scale", "scaling", "schema", "security", "server",
    "service", "sorting", "sql", "stack", "string", "system", "test", "testing", "thread",
    "threads", "throughput", "transaction", "tree", "type", "variable", "code", "coding",
    "program", "programming", "software", "architecture", "framework", "library", "loop",
    "map", "dictionary", "tuple", "generator", "decorator", "closure", "exception", "mutex",
    "deadlock", "race", "load", "balancer", "sharding", "replication", "kafka", "grpc",
    "tcp", "udp", "dns", "url", "browser", "frontend", "backend", "javascript", "java",
    "golang", "rust", "c++", "react", "git", "ci", "cd", "pipeline", "cloud", "aws",
    "idempotent", "idempotency", "oauth", "authentication", "encryption", "o(n)", "runtime",
    "compile", "garbage", "collection", "virtual", "machine", "kernel", "os", "cpu",
}


@dataclass
class DetectorConfig:
    min_words: int = 4
    require_technical: bool = True
    similarity: float = 0.85
    history: int = 30


class QuestionDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()
        self._recent: deque[list[str]] = deque(maxlen=self.cfg.history)

    @staticmethod
    def is_question_form(text: str) -> bool:
        t = text.strip().lower()
        return t.endswith("?") or bool(INTERROGATIVE.match(t))

    @staticmethod
    def is_technical(words: list[str]) -> bool:
        return any(w in TECH_TERMS for w in words)

    def is_duplicate(self, words: list[str]) -> bool:
        return any(difflib.SequenceMatcher(a=words, b=prev, autojunk=False).ratio()
                   >= self.cfg.similarity for prev in self._recent)

    def check(self, text: str) -> str | None:
        """Return a reason string if `text` should be asked, else None. Records it if so."""
        words = normalize_text(text).split()
        if len(words) < self.cfg.min_words or not self.is_question_form(text):
            return None
        if self.cfg.require_technical and not self.is_technical(words):
            return None
        if self.is_duplicate(words):
            return None
        self._recent.append(words)
        return "question"

    def remember(self, text: str) -> None:
        """Record a manually asked question so it is not detected again."""
        self._recent.append(normalize_text(text).split())
