"""Decide which complete utterances are interview questions (technical or behavioural).

Input is *utterances*: finals already merged into complete sentences (asr/sentences.py).
Partials and fragments are never considered, so incomplete transcripts are not sent.

Heuristics (no model call):
  * at least ``min_words`` words
  * a question form: ends with "?", starts with an interrogative ("how would you ..."),
    or an interview prompt ("explain ...", "walk me through ...", "design ...")
  * optionally (``filter_small_talk``) an interview cue: a technical term, or a behavioural
    cue ("tell me about yourself", "why did you choose ...", "describe a time ...",
    strengths, experience, team, ...). Small talk such as "How are you doing today?" or
    "Can you hear me?" is ignored.
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


BEHAVIOURAL = re.compile(
    r"\b(tell (me|us) about|about yourself|introduce yourself|walk me through your|"
    r"why (did|do|would|should|are|were) (you|we)|why (this|our|us|here)|what made you|"
    r"what (is|are|was|were|would be) your|what's your|whats your|what have you|"
    r"what do you know about|where do you see yourself|describe (a|an|the) (time|situation|project)|"
    r"give (me|us) an example|a time when|how (do|did|would) you (handle|deal|manage|"
    r"prioriti[sz]e|work|approach|learn|stay|motivate|resolve|cope)|"
    r"do you have (any )?questions|how would your (colleagues|manager|team))\b"
)

BEHAVIOURAL_TERMS = {
    "strength", "strengths", "weakness", "weaknesses", "experience", "project", "projects",
    "team", "teammate", "conflict", "challenge", "challenging", "mistake", "failure",
    "achievement", "accomplishment", "goal", "goals", "career", "motivation", "motivates",
    "salary", "role", "position", "company", "background", "studied", "study", "major",
    "school", "university", "college", "degree", "manager", "leadership", "lead", "deadline",
    "pressure", "feedback", "hire", "job", "internship", "intern", "passion", "proud",
    "difficult", "colleague", "customer", "stakeholder", "decision", "skills", "hobbies",
}

SMALL_TALK = re.compile(
    r"\b(how are you|how's it going|how is it going|how have you been|can you hear me|"
    r"can you see (me|my screen)|is (that|this) (ok|okay|alright|fine)|does that make sense|"
    r"what are you doing|are you there|are you ready|shall we (start|begin)|"
    r"nice to meet you|how was your (day|weekend))\b"
)


@dataclass
class DetectorConfig:
    min_words: int = 4
    filter_small_talk: bool = True
    similarity: float = 0.85
    history: int = 30


class QuestionDetector:
    def __init__(self, config: DetectorConfig | None = None) -> None:
        self.cfg = config or DetectorConfig()
        self._recent: deque[list[str]] = deque(maxlen=self.cfg.history)

    @staticmethod
    def is_question_form(text: str) -> bool:
        t = text.strip().lower()
        return t.endswith("?") or bool(INTERROGATIVE.match(t)) or "?" in t

    @staticmethod
    def is_technical(words: list[str]) -> bool:
        return any(w in TECH_TERMS for w in words)

    @staticmethod
    def is_behavioural(norm: str, words: list[str]) -> bool:
        return bool(BEHAVIOURAL.search(norm)) or any(w in BEHAVIOURAL_TERMS for w in words)

    def is_interview_question(self, text: str) -> bool:
        norm = normalize_text(text)
        words = norm.split()
        if self.is_technical(words) or self.is_behavioural(norm, words):
            return True
        return False

    def is_duplicate(self, words: list[str]) -> bool:
        return any(difflib.SequenceMatcher(a=words, b=prev, autojunk=False).ratio()
                   >= self.cfg.similarity for prev in self._recent)

    def check(self, text: str) -> str | None:
        """Return a reason string if `text` should be asked, else None. Records it if so."""
        norm = normalize_text(text)
        words = norm.split()
        if len(words) < self.cfg.min_words or not self.is_question_form(text):
            return None
        if self.cfg.filter_small_talk:
            if SMALL_TALK.search(norm) and not self.is_technical(words):
                return None
            if not self.is_interview_question(text):
                return None
        if self.is_duplicate(words):
            return None
        self._recent.append(words)
        return "question"

    def remember(self, text: str) -> None:
        """Record a manually asked question so it is not detected again."""
        self._recent.append(normalize_text(text).split())
