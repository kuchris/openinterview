"""LLM providers for practice answers and answer feedback.

OpenCodeProvider talks to a locally running `opencode serve` through its documented HTTP
API (verified against OpenCode 1.19.3's OpenAPI spec at /doc):

  POST /session?directory=D            {"title", "permission": [...]}       -> Session
  POST /session/{id}/message?directory {"agent", "system", "format", "parts"} -> {info, parts}

Safety: the session is created with every tool permission denied and uses the read-only
"plan" agent, in an empty working directory, so the model can only answer, never read or
change files or run commands. One OpenCode session is kept per practice run, so follow-up
questions keep their context. Answers use OpenCode's structured output
(format=json_schema), returned in info.structured.

OpenAICompatibleProvider is a lighter alternative for any /v1/chat/completions server.
"""

from __future__ import annotations

import json
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Protocol

SYSTEM_PROMPT = (
    "You are a concise technical interview coach helping someone practise. For each "
    "question, give a short spoken-style answer in English (2-4 sentences), 3 to 5 key "
    "technical points a strong answer must cover, and a short Python example only when code "
    "genuinely helps (otherwise an empty string). Be accurate; do not pad."
)

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "short_answer": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}, "minItems": 3,
                       "maxItems": 5},
        "python_example": {"type": "string"},
    },
    "required": ["short_answer", "key_points", "python_example"],
    "additionalProperties": False,
}

FEEDBACK_SCHEMA = {
    "type": "object",
    "properties": {
        "covered": {"type": "array", "items": {"type": "string"}},
        "missed": {"type": "array", "items": {"type": "string"}},
        "tip": {"type": "string"},
    },
    "required": ["covered", "missed", "tip"],
    "additionalProperties": False,
}


@dataclass
class Answer:
    short_answer: str = ""
    key_points: list[str] = field(default_factory=list)
    python_example: str = ""
    model: str = ""
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Feedback:
    covered: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)
    tip: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class ProviderError(RuntimeError):
    pass


class AnswerProvider(Protocol):
    name: str

    def health(self) -> tuple[bool, str]: ...
    def answer(self, question: str, context: list[str]) -> Answer: ...
    def grade(self, question: str, answer: Answer, spoken: str) -> Feedback: ...


def _question_prompt(question: str, context: list[str]) -> str:
    ctx = "\n".join(context[-8:])
    return (f"Recent conversation (for context, may contain ASR errors):\n{ctx}\n\n"
            f"Interview question: {question}" if ctx else f"Interview question: {question}")


def _grade_prompt(question: str, answer: Answer, spoken: str) -> str:
    points = "\n".join(f"- {p}" for p in answer.key_points)
    return (f"Interview question: {question}\n\nKey points of a strong answer:\n{points}\n\n"
            f"The candidate answered (speech-to-text, may contain recognition errors):\n"
            f"{spoken or '(no answer was spoken)'}\n\nWhich key points did they cover and "
            f"which did they miss? Give one short, specific tip.")


def _parse_json_text(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{"):]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ProviderError("model did not return JSON")
    return json.loads(text[start:end + 1])


def _to_answer(data: dict, model: str, seconds: float) -> Answer:
    points = [str(p) for p in data.get("key_points", [])][:5]
    return Answer(str(data.get("short_answer", "")).strip(), points,
                  str(data.get("python_example", "") or "").strip(), model, round(seconds, 2))


def _to_feedback(data: dict) -> Feedback:
    return Feedback([str(x) for x in data.get("covered", [])],
                    [str(x) for x in data.get("missed", [])], str(data.get("tip", "")))


class _HTTP:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, body: dict | None = None,
                query: dict | None = None, headers: dict | None = None) -> dict:
        url = self.base_url + path + ("?" + urllib.parse.urlencode(query) if query else "")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise ProviderError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ProviderError(f"cannot reach {self.base_url}: {exc}") from exc


class OpenCodeProvider:
    name = "opencode"

    def __init__(self, base_url: str = "http://127.0.0.1:4096", model: str | None = None,
                 agent: str = "plan", timeout: float = 60.0, directory: str | None = None) -> None:
        self.http = _HTTP(base_url, timeout)
        self.model = model  # "providerID/modelID"; None = OpenCode's configured default
        self.agent = agent
        # An empty scratch directory: OpenCode treats it as the project root.
        self.directory = directory or tempfile.mkdtemp(prefix="openinterview-opencode-")
        self.session_id: str | None = None

    def _q(self) -> dict:
        return {"directory": self.directory}

    def health(self) -> tuple[bool, str]:
        try:
            h = self.http.request("GET", "/global/health")
            return bool(h.get("healthy")), f"OpenCode {h.get('version', '?')}"
        except ProviderError as exc:
            return False, str(exc)

    def _session(self) -> str:
        if self.session_id is None:
            s = self.http.request("POST", "/session", {
                "title": "OpenInterview practice",
                "permission": [{"permission": "*", "pattern": "*", "action": "deny"}],
            }, self._q())
            self.session_id = s["id"]
        return self.session_id

    def _ask(self, text: str, schema: dict) -> tuple[dict, str]:
        body: dict = {"agent": self.agent, "system": SYSTEM_PROMPT,
                      "format": {"type": "json_schema", "schema": schema},
                      "parts": [{"type": "text", "text": text}]}
        if self.model and "/" in self.model:
            provider, model = self.model.split("/", 1)
            body["model"] = {"providerID": provider, "modelID": model}
        r = self.http.request("POST", f"/session/{self._session()}/message", body, self._q())
        info = r.get("info", {})
        if info.get("error"):
            err = info["error"]
            msg = (err.get("data") or {}).get("message") or err.get("name") or "error"
            raise ProviderError(f"OpenCode: {msg}")
        model = f"{info.get('providerID', '')}/{info.get('modelID', '')}".strip("/")
        if isinstance(info.get("structured"), dict):
            return info["structured"], model
        text = "".join(p.get("text", "") for p in r.get("parts", []) if p.get("type") == "text")
        return _parse_json_text(text), model

    def answer(self, question: str, context: list[str]) -> Answer:
        t0 = time.monotonic()
        data, model = self._ask(_question_prompt(question, context), ANSWER_SCHEMA)
        return _to_answer(data, model, time.monotonic() - t0)

    def grade(self, question: str, answer: Answer, spoken: str) -> Feedback:
        data, _ = self._ask(_grade_prompt(question, answer, spoken), FEEDBACK_SCHEMA)
        return _to_feedback(data)

    def close(self) -> None:
        if self.session_id:
            try:
                self.http.request("DELETE", f"/session/{self.session_id}", query=self._q())
            except ProviderError:
                pass
            self.session_id = None


class OpenAICompatibleProvider:
    """Any OpenAI-compatible /v1/chat/completions server (e.g. a local model server)."""

    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 60.0) -> None:
        self.http = _HTTP(base_url, timeout)
        self.model = model
        self.api_key = api_key
        self.history: list[dict] = []  # conversation context across questions

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def health(self) -> tuple[bool, str]:
        try:
            self.http.request("GET", "/models", headers=self._headers())
            return True, f"{self.http.base_url} ({self.model})"
        except ProviderError as exc:
            return False, str(exc)

    def _chat(self, text: str, schema: dict) -> dict:
        messages = [{"role": "system", "content": SYSTEM_PROMPT + " Reply with only a JSON "
                     f"object matching this JSON schema: {json.dumps(schema)}"},
                    *self.history[-8:], {"role": "user", "content": text}]
        r = self.http.request("POST", "/chat/completions", {
            "model": self.model, "messages": messages, "temperature": 0.2,
            "response_format": {"type": "json_object"}}, headers=self._headers())
        content = r["choices"][0]["message"]["content"]
        self.history += [{"role": "user", "content": text},
                         {"role": "assistant", "content": content}]
        return _parse_json_text(content)

    def answer(self, question: str, context: list[str]) -> Answer:
        t0 = time.monotonic()
        data = self._chat(_question_prompt(question, context), ANSWER_SCHEMA)
        return _to_answer(data, self.model, time.monotonic() - t0)

    def grade(self, question: str, answer: Answer, spoken: str) -> Feedback:
        return _to_feedback(self._chat(_grade_prompt(question, answer, spoken), FEEDBACK_SCHEMA))

    def close(self) -> None:
        pass
