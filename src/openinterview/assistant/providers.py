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
import re
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
    'Format: {"short_answer": string, "key_points": [string], "python_example": string, '
    '"gap": string}. short_answer: what a strong candidate would say out loud, 1-2 '
    "sentences, at most 45 words, plain spoken English, ready to say as-is: never "
    "[placeholders], brackets or blanks to fill in. key_points: 3 to 5 terse cues to "
    "remember, each at most 8 words, not full sentences. python_example: only for technical "
    "questions where code genuinely helps (algorithms, data structures, language features), "
    'at most 10 runnable lines; otherwise "". Behavioural questions (about the candidate, '
    "their background, motivation, teamwork, strengths...): write a first-person model "
    "answer from facts in the candidate profile. When the profile lacks the specific thing "
    "asked (a conflict story, a weakness, why this company...), still give an honest answer "
    "they can say: use the closest real facts from the profile, and how they approach such "
    "situations, without inventing events, employers, dates, numbers, outcomes or "
    "feelings. For motivation questions use what the role and job description offer that "
    "matches their real background. Key points outline the structure (e.g. STAR); no code; "
    "they are cues for what to say, never notes about the profile or what is missing. "
    'gap: "" when the profile fully supports the answer; otherwise one short phrase naming '
    "the missing fact, at most 10 words (e.g. 'No real conflict story in your profile')."
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


def language_rules(code: str | None) -> str:
    """For interviews not in English: answer in that language, in a style to say aloud."""
    from ..languages import get

    if not code or code == "en":
        return ""
    lang = get(code)
    return (f"\n\nThis interview is in {lang.name}. Questions arrive in {lang.name} (speech "
            f"recognition may mix in English words). Write short_answer and key_points in "
            f"{lang.answer_style}; keep them as short as the rules say. Write gap in English. "
            "python_example stays code.")


def coach_prompt(profile: str = "", target: dict | None = None) -> str:
    """The coach's system prompt, with the candidate's profile, this interview's target and
    its language (target["language"]) appended when present."""
    lang = language_rules((target or {}).get("language"))
    if not profile.strip():
        return COACH_PROMPT + target_text(target) + lang
    return (COACH_PROMPT + target_text(target) + lang + "\n\nCandidate profile (facts about the person you are coaching; "
            "the only source of truth about them):\n<<<\n" + profile.strip() + "\n>>>\n"
            "Rules for using the profile: state only facts that are written in it, without "
            "embellishing (no added motives, feelings, anecdotes or numbers). If the profile "
            "does not contain what the question asks for (e.g. a weakness, a conflict story, "
            "a reason for a choice), do NOT make it up: answer honestly from the closest real "
            "facts and their general approach, with no placeholders, and name the missing "
            "fact in gap. For technical questions, mention their experience only when "
            "clearly relevant.")

PREP_PROMPT = (
    "You help a candidate prepare for a specific job interview. You compare their profile "
    "with the interview and find the likely questions their profile cannot yet answer "
    "well. Always reply with ONLY a JSON object in the exact format requested: no prose, "
    "no code fences."
)

PREP_RULES = (
    'Format: {"gaps": [{"question": string, "missing": string, "hint": string}]}. List 3 '
    "to 5 interview questions that are likely for this role and company (behavioural, "
    "motivation, or a key skill from the job description) where the profile lacks the facts "
    "for a strong, true answer. question: as an interviewer would ask it. missing: what "
    "the profile lacks, at most 12 words. hint: what to write, at most 20 words (e.g. "
    "'One real disagreement: who, what you did, the result'). Skip anything the profile "
    "already covers well. Never assume facts about the candidate."
)

PREP_SCHEMA = {
    "type": "object",
    "properties": {"gaps": {"type": "array", "items": {"type": "object"}}},
    "required": ["gaps"],
}


def _prep_request(profile: str, target: dict | None) -> str:
    return ((target_text(target).strip() or "This interview: (no company or role given)")
            + "\n\nCandidate profile:\n<<<\n" + (profile.strip() or "(empty)") + "\n>>>")


def _to_gaps(data: dict) -> list[dict]:
    out = []
    for g in data.get("gaps", [])[:6]:
        if isinstance(g, dict) and str(g.get("question", "")).strip():
            out.append({k: str(g.get(k, "")).strip() for k in ("question", "missing", "hint")})
    return out

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
        "gap": {"type": "string"},
    },
    "required": ["short_answer", "key_points", "python_example", "gap"],
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
    gap: str = ""  # what the profile lacks for this answer ("" = nothing)

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
    if "gaps" in schema["properties"]:
        return PREP_RULES
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


# Fill-in-the-blank placeholders the model may still produce, e.g. "[your weakness]" or
# "[Add your reason for Acme here.]". Code such as "nums[i]" is left alone.
PLACEHOLDER = re.compile(r"\s*\[(?:add|your|insert|specific|fill|to add|name of|reason|"
                         r"example|describe|mention)\b[^\]]{0,120}\]", re.I)


PROFILE_NOTE = re.compile(r"\b(in|to|from) (your |the )?profile\b", re.I)


def _strip_placeholders(text: str) -> tuple[str, list[str]]:
    found = [m.group(0).strip(" []") for m in PLACEHOLDER.finditer(text)]
    return PLACEHOLDER.sub("", text).strip(), found


def _to_answer(data: dict, model: str, seconds: float) -> Answer:
    short, found = _strip_placeholders(str(data.get("short_answer", "")))
    points = []
    for p in [str(p) for p in data.get("key_points", [])][:5]:
        p, f = _strip_placeholders(p)
        found += f
        if not p.strip(" :-") or (f and len(p.split()) <= 2):  # drop "Add:" stubs
            continue
        if PROFILE_NOTE.search(p):  # "No conflict story in profile": that's the gap's job
            continue
        points.append(p)
    gap = str(data.get("gap", "") or "").strip()
    if not gap and found:
        gap = found[0][:80]
    return Answer(short, points, str(data.get("python_example", "") or "").strip(), model,
                  round(seconds, 2), gap=gap)


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
    PREP = "interview-prep"

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
            self.PREP: {
                "description": "Finds profile gaps for an interview (OpenInterview)",
                "mode": "primary", "prompt": PREP_PROMPT, "temperature": 0.2,
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

    def _one_off(self, title: str, text: str, schema: dict, agent: str,
                 timeout: float) -> dict:
        """A request in its own throwaway session, outside the practice conversation."""
        sid = self._new_session(title)
        try:
            data, _ = self._ask(text, schema, agent=agent, session=sid, timeout=timeout)
        finally:
            try:
                self.http.request("DELETE", f"/session/{sid}", query=self._q())
            except ProviderError:
                pass
        return data

    def build_profile(self, cv_text: str, existing: str = "") -> tuple[str, list[str]]:
        """Organise a CV into a profile (throwaway session: the CV stays out of practice)."""
        return _to_profile(self._one_off("OpenInterview profile import",
                                         _profile_request(cv_text, existing), PROFILE_SCHEMA,
                                         self.BUILDER, max(PROFILE_TIMEOUT, self.http.timeout)))

    def check_gaps(self, profile: str, target: dict | None) -> list[dict]:
        """Likely questions for this interview that the profile cannot answer yet."""
        return _to_gaps(self._one_off("OpenInterview prep check", _prep_request(profile, target),
                                      PREP_SCHEMA, self.PREP, max(120.0, self.http.timeout)))

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

    def check_gaps(self, profile: str, target: dict | None) -> list[dict]:
        messages = [{"role": "system", "content": PREP_PROMPT + "\n" + PREP_RULES},
                    {"role": "user", "content": _prep_request(profile, target)}]
        r = self.http.request("POST", "/chat/completions", {
            "model": self.model, "messages": messages, "temperature": 0.2,
            "response_format": {"type": "json_object"}}, headers=self._headers(),
            timeout=max(120.0, self.http.timeout))
        return _to_gaps(_parse_json_text(r["choices"][0]["message"]["content"]))

    def close(self) -> None:
        pass
