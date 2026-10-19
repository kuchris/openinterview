"""LLM providers for practice answers and answer feedback.

OpenCodeProvider talks to a locally running `opencode serve` through its documented HTTP
API (verified against OpenCode 1.19.3's OpenAPI spec at /doc):

  POST /session?directory=D            {"title", "permission": [...]}       -> Session
  POST /session/{id}/message?directory {"agent", "system", "model", "parts"}  -> {info, parts}

Agent: the provider works in its own empty directory and writes an `opencode.json` there
defining a dedicated "interview-coach" agent: a short prompt, all tools off, low
temperature. OpenCode's built-in agents carry ~6.5k tokens of coding-agent instructions,
which made answers slow and verbose; the coach agent sends ~300 tokens per question.

Safety: tools are disabled on the agent, the session denies every tool permission, and the
working directory is empty, so the model can only answer, never read or change files or run
commands. One OpenCode session is kept per practice run, so follow-ups keep context.

Output: the model is asked for a JSON object in its text reply, which is parsed (with one
retry). OpenCode's own structured output (format=json_schema) is *not* used: OpenCode
implements it as a "StructuredOutput" tool, which the deny-all permission hides, and some
models (e.g. DeepSeek V4.1 Flash) then fail with StructuredOutputError.

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

COACH_PROMPT = (
    "You coach someone preparing for a job interview, technical or behavioural. Always reply "
    "with ONLY a JSON object in the exact format requested: no prose, no markdown, no code "
    "fences. Be accurate and concise; no filler or caveats. Questions come from speech "
    "recognition and may be garbled: answer the most likely intended question. If several "
    "questions are asked at once, answer them together."
)

ANSWER_RULES = (
    'Format: {"short_answer": string, "key_points": [string], "python_example": string}. '
    "short_answer: what a strong candidate would say out loud, 1-2 sentences, at most 45 "
    "words, plain spoken English. key_points: 3 to 5 terse cues to remember, each at most 8 "
    "words, not full sentences. python_example: only for technical questions where code "
    'genuinely helps (algorithms, data structures, language features), at most 10 runnable '
    'lines; otherwise "". Behavioural questions (about the candidate, their background, '
    "motivation, teamwork, strengths...): write a first-person model answer. Use facts from "
    "the candidate profile when one is given; use short [placeholders] (e.g. [your degree]) "
    "for anything the profile does not cover. Never invent employers, dates, numbers or "
    "achievements. Key points outline the structure (e.g. STAR: situation, task, action, "
    "result); no code."
)


def target_text(target: dict | None) -> str:
    """The interview this session is for (company, role, job description excerpt)."""
    if not target:
        return ""
    parts = []
    if target.get("company"):
        parts.append(f"Company: {target['company']}")
    if target.get("role"):
        parts.append(f"Role: {target['role']}")
    if target.get("job_description"):
        parts.append("Job description:\n" + target["job_description"].strip()[:4000])
    if not parts:
        return ""
    return ("\n\nThis interview (tailor answers to it: stress the experience, skills and "
            "motivations that match; use only facts stated here about the company, never "
            "guess others):\n<<<\n" + "\n".join(parts) + "\n>>>")


def coach_prompt(profile: str = "", target: dict | None = None) -> str:
    """The coach's system prompt, with the candidate's profile and this interview's target
    appended when present."""
    if not profile.strip():
        return COACH_PROMPT + target_text(target)
    return (COACH_PROMPT + target_text(target) + "\n\nCandidate profile (facts about the person you are coaching; "
            "the only source of truth about them):\n<<<\n" + profile.strip() + "\n>>>\n"
            "Rules for using the profile: state only facts that are written in it, without "
            "embellishing (no added motives, feelings, anecdotes or numbers). If the profile "
            "does not contain what the question asks for (e.g. a weakness, a conflict story, "
            "a reason for a choice), do NOT make it up: answer with short [placeholders] such "
            "as [your weakness] and make one key point say what to add to the profile. For "
            "technical questions, mention their experience only when clearly relevant.")

FEEDBACK_RULES = (
    'Format: {"covered": [string], "missed": [string], "tip": string}. Compare the '
    "candidate's spoken answer with the key points. covered / missed: key points as terse "
    "phrases of at most 8 words; be fair to paraphrases and speech-recognition errors. tip: "
    "one concrete sentence, at most 25 words, on what to add or say better."
)

PROFILE_BUILDER_PROMPT = (
    "You turn a candidate's CV into a concise interview-preparation profile in Markdown. "
    "Use ONLY facts stated in the CV text and in their existing profile notes. Never invent "
    "or guess employers, dates, numbers, motives, feelings or outcomes. Always reply with "
    "ONLY a JSON object in the exact format requested: no prose, no code fences."
)

PROFILE_RULES = (
    'Format: {"profile_markdown": string, "to_add": [string]}. profile_markdown: at most '
    "600 words, with exactly these sections: '## About me' (name, current role/status, "
    "location if given), '## Target role' ('- [to add]' unless the sources state one), "
    "'## Experience' (each role: title, organisation, dates, then 1-2 bullets of concrete "
    "work and impact, keeping the CV's numbers), '## Projects', '## Education', "
    "'## Skills' (grouped, one line each), '## Strengths' (2-3, each backed by evidence "
    "from the CV), '## Weaknesses' ('- [to add]' unless stated), '## Stories (STAR)' (2-3 "
    "outlines drafted from CV achievements: Situation / Task / Action / Result using only "
    "CV facts, writing [?] for any part the CV does not state). Keep everything from the "
    "existing notes. Use '- [to add]' for any section the sources do not cover. to_add: 3 "
    "to 6 short questions the candidate should answer to complete the profile (e.g. "
    "'Which role and company are you targeting?', 'What is a real weakness and how are "
    "you improving it?')."
)

PROFILE_SCHEMA = {
    "type": "object",
    "properties": {"profile_markdown": {"type": "string"},
                   "to_add": {"type": "array", "items": {"type": "string"}}},
    "required": ["profile_markdown", "to_add"],
}


PROFILE_TIMEOUT = 180.0  # organising a long CV can take a minute or more


def _profile_request(cv_text: str, existing: str) -> str:
    notes = existing.strip() or "(none)"
    return (f"Existing profile notes (keep them):\n<<<\n{notes}\n>>>\n\n"
            f"CV text:\n<<<\n{cv_text.strip()[:20000]}\n>>>")


def _to_profile(data: dict) -> tuple[str, list[str]]:
    md = str(data.get("profile_markdown", "")).strip()
    if not md:
        raise ProviderError("model returned an empty profile")
    return md, [str(q) for q in data.get("to_add", [])][:8]


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


def _rules(schema: dict) -> str:
    if "profile_markdown" in schema["properties"]:
        return PROFILE_RULES
    return FEEDBACK_RULES if "covered" in schema["properties"] else ANSWER_RULES


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
                query: dict | None = None, headers: dict | None = None,
                timeout: float | None = None) -> dict:
        url = self.base_url + path + ("?" + urllib.parse.urlencode(query) if query else "")
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json",
                                              **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise ProviderError(f"HTTP {exc.code} from {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            raise ProviderError(f"cannot reach {self.base_url}: {exc}") from exc


class OpenCodeProvider:
    name = "opencode"

    AGENT = "interview-coach"
    BUILDER = "profile-builder"

    def __init__(self, base_url: str = "http://127.0.0.1:4096", model: str | None = None,
                 timeout: float = 60.0, directory: str | None = None, profile: str = "") -> None:
        self.http = _HTTP(base_url, timeout)
        self.model = model  # "providerID/modelID"; None = OpenCode's configured default
        self.profile = profile
        self.target: dict = {}
        # An empty scratch directory: OpenCode treats it as the project root, and loads the
        # coach agent from its opencode.json.
        self.directory = directory or tempfile.mkdtemp(prefix="openinterview-opencode-")
        self._write_agent_config()
        self.session_id: str | None = None

    def set_profile(self, profile: str) -> None:
        self.set_context(profile, self.target)

    def set_context(self, profile: str, target: dict | None = None) -> None:
        """Use a new profile / interview target with a fresh conversation. OpenCode caches a
        directory's config, so this starts a new directory and session (the old session is
        deleted)."""
        self.close()
        self.profile = profile
        self.target = dict(target or {})
        self.directory = tempfile.mkdtemp(prefix="openinterview-opencode-")
        self._write_agent_config()

    def _write_agent_config(self) -> None:
        import os

        config = {"$schema": "https://opencode.ai/config.json", "agent": {
            self.AGENT: {
                "description": "Concise interview coach (OpenInterview)",
                "mode": "primary", "prompt": coach_prompt(self.profile, self.target),
                "temperature": 0.3,
                "tools": {"*": False},
            },
            self.BUILDER: {
                "description": "Turns a CV into an interview profile (OpenInterview)",
                "mode": "primary", "prompt": PROFILE_BUILDER_PROMPT, "temperature": 0.1,
                "tools": {"*": False},
            },
        }}
        with open(os.path.join(self.directory, "opencode.json"), "w") as f:
            json.dump(config, f, indent=2)

    def _q(self) -> dict:
        return {"directory": self.directory}

    def health(self) -> tuple[bool, str]:
        try:
            h = self.http.request("GET", "/global/health")
            return bool(h.get("healthy")), f"OpenCode {h.get('version', '?')}"
        except ProviderError as exc:
            return False, str(exc)

    def _new_session(self, title: str) -> str:
        s = self.http.request("POST", "/session", {
            "title": title,
            "permission": [{"permission": "*", "pattern": "*", "action": "deny"}],
        }, self._q())
        return s["id"]

    def _session(self) -> str:
        if self.session_id is None:
            self.session_id = self._new_session("OpenInterview practice")
        return self.session_id

    def _send(self, text: str, schema: dict, agent: str | None = None,
              session: str | None = None, timeout: float | None = None) -> tuple[dict, str]:
        body: dict = {"agent": agent or self.AGENT, "system": _rules(schema),
                      "parts": [{"type": "text", "text": text}]}
        if self.model and "/" in self.model:
            provider, model = self.model.split("/", 1)
            body["model"] = {"providerID": provider, "modelID": model}
        sid = session or self._session()
        r = self.http.request("POST", f"/session/{sid}/message", body, self._q(),
                              timeout=timeout)
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

    def _ask(self, text: str, schema: dict, agent: str | None = None,
             session: str | None = None, timeout: float | None = None) -> tuple[dict, str]:
        try:
            return self._send(text, schema, agent, session, timeout)
        except (ProviderError, ValueError) as exc:
            if isinstance(exc, ProviderError) and not str(exc).startswith("model did not"):
                raise  # transport / auth errors: retrying will not help
            return self._send("Your previous reply was not valid JSON. " + text, schema,
                              agent, session, timeout)

    def build_profile(self, cv_text: str, existing: str = "") -> tuple[str, list[str]]:
        """Organise a CV into a profile. Uses its own throwaway session, so the CV does not
        land in the practice conversation."""
        sid = self._new_session("OpenInterview profile import")
        try:
            data, _ = self._ask(_profile_request(cv_text, existing), PROFILE_SCHEMA,
                                agent=self.BUILDER, session=sid,
                                timeout=max(PROFILE_TIMEOUT, self.http.timeout))
        finally:
            try:
                self.http.request("DELETE", f"/session/{sid}", query=self._q())
            except ProviderError:
                pass
        return _to_profile(data)

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

    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 60.0,
                 profile: str = "") -> None:
        self.http = _HTTP(base_url, timeout)
        self.model = model
        self.api_key = api_key
        self.profile = profile
        self.target: dict = {}
        self.history: list[dict] = []  # conversation context across questions

    def set_profile(self, profile: str) -> None:
        self.set_context(profile, self.target)

    def set_context(self, profile: str, target: dict | None = None) -> None:
        self.profile = profile
        self.target = dict(target or {})
        self.history = []

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def health(self) -> tuple[bool, str]:
        try:
            self.http.request("GET", "/models", headers=self._headers())
            return True, f"{self.http.base_url} ({self.model})"
        except ProviderError as exc:
            return False, str(exc)

    def _chat(self, text: str, schema: dict) -> dict:
        messages = [{"role": "system",
                     "content": coach_prompt(self.profile, self.target) + "\n" + _rules(schema)},
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

    def build_profile(self, cv_text: str, existing: str = "") -> tuple[str, list[str]]:
        messages = [{"role": "system", "content": PROFILE_BUILDER_PROMPT + "\n" + PROFILE_RULES},
                    {"role": "user", "content": _profile_request(cv_text, existing)}]
        r = self.http.request("POST", "/chat/completions", {
            "model": self.model, "messages": messages, "temperature": 0.1,
            "response_format": {"type": "json_object"}}, headers=self._headers(),
            timeout=max(PROFILE_TIMEOUT, self.http.timeout))
        return _to_profile(_parse_json_text(r["choices"][0]["message"]["content"]))

    def close(self) -> None:
        pass
