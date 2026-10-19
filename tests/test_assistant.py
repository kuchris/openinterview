"""Practice assistant: detection, OpenCode protocol (mock server), practice flow, API."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from openinterview.assistant.detector import QuestionDetector
from openinterview.assistant.practice import PracticeAssistant
from openinterview.assistant.providers import (Answer, Feedback, OpenAICompatibleProvider,
                                               OpenCodeProvider, ProviderError)
from openinterview.config import AssistantConfig

# -- detector ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Can you explain the difference between a process and a thread?",
    "How would you design a rate limiter for a REST API",
    "So, walk me through how a hash map handles collisions.",
    "Explain how the GIL affects multithreading in Python.",
    "What is the time complexity of binary search?",
])
def test_detects_technical_questions(text):
    assert QuestionDetector().check(text)


@pytest.mark.parametrize("text", [
    "Let's start with tell me about yourself question. Who are you? What have you studied?",
    "Okay, so why did you choose that particular major and at that particular school?",
    "What is your greatest weakness?",
    "Describe a time you had a conflict with a teammate.",
    "Where do you see yourself in five years?",
])
def test_detects_behavioural_questions(text):
    assert QuestionDetector().check(text)


@pytest.mark.parametrize("text", [
    "How are you doing today?",                       # small talk
    "Can you hear me okay?",                          # small talk
    "What are you doing?",                            # not an interview question
    "I'm going to lead with questions, but feel free to ask me about the role.",  # statement
    "Thanks, that makes sense.",                      # not a question
    "Why?",                                           # too short
    "I think we should use a hash map here.",         # statement
])
def test_ignores_non_questions(text):
    assert not QuestionDetector().check(text)


def test_near_duplicate_question_not_asked_twice():
    d = QuestionDetector()
    assert d.check("How would you design a rate limiter for a REST API?")
    assert not d.check("how would you design a rate limiter for a rest api")
    assert d.check("How would you design a URL shortener service?")


# -- mock OpenCode server (shapes from OpenCode 1.19.3's /doc OpenAPI) -----------------------

class MockOpenCode:
    def __init__(self, mode="text"):
        self.mode = mode
        self.calls = 0
        self.requests = []
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                mock.requests.append(("GET", self.path, None))
                if self.path.startswith("/global/health"):
                    return self._send(200, {"healthy": True, "version": "1.19.3"})
                self._send(404, {})

            def do_DELETE(self):
                mock.requests.append(("DELETE", self.path, None))
                self._send(200, True)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
                mock.requests.append(("POST", self.path, body))
                if self.path.startswith("/session?"):
                    return self._send(200, {"id": "ses_test123", "title": body.get("title")})
                if "/message" in self.path:
                    return self._send(200, mock.reply(body))
                self._send(404, {})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def reply(self, body):
        self.calls += 1
        info = {"id": "msg_1", "sessionID": "ses_test123", "role": "assistant",
                "providerID": "opencode-go", "modelID": "test-model", "time": {"created": 1}}
        grading = '"covered"' in body["system"]
        if self.mode == "expired":
            info["error"] = {"name": "APIError", "data": {"message": "This API key has expired."}}
            return {"info": info, "parts": []}
        if self.mode == "garbage_once" and self.calls == 1:
            return {"info": info, "parts": [{"type": "text", "text": "Sure! A mutex is..."}]}
        if grading:
            data = {"covered": ["memory isolation"], "missed": ["context switch cost"],
                    "tip": "Mention shared memory between threads."}
        else:
            data = {"short_answer": "A process has its own memory; threads share it.",
                    "key_points": ["isolation", "shared memory", "context switch cost"],
                    "python_example": "import threading"}
        if self.mode == "structured":  # server-side structured output, if a model provides it
            info["structured"] = data
            return {"info": info, "parts": []}
        return {"info": info, "parts": [{"type": "text", "text": "```json\n"
                                         + json.dumps(data) + "\n```"}]}

    def close(self):
        self.server.shutdown()


@pytest.fixture
def mock_opencode():
    m = MockOpenCode()
    yield m
    m.close()


def test_opencode_request_shape_and_structured_answer(mock_opencode, tmp_path):
    p = OpenCodeProvider(mock_opencode.url, model="opencode-go/test-model",
                         directory=str(tmp_path))
    assert p.health() == (True, "OpenCode 1.19.3")
    a = p.answer("What is the difference between a process and a thread?", ["Them: hi"])
    assert a.short_answer.startswith("A process") and len(a.key_points) == 3
    assert a.model == "opencode-go/test-model"

    method, path, create = mock_opencode.requests[1]
    assert (method, path.split("?")[0]) == ("POST", "/session")
    assert f"directory={tmp_path}".replace("/", "%2F") in path
    assert create["permission"] == [{"permission": "*", "pattern": "*", "action": "deny"}]
    _, path, msg = mock_opencode.requests[2]
    assert path.startswith("/session/ses_test123/message")
    assert msg["agent"] == "interview-coach"
    # No OpenCode structured output: it is a tool, hidden by the deny-all permission.
    assert "format" not in msg and '"short_answer"' in msg["system"]
    agent_cfg = json.loads((tmp_path / "opencode.json").read_text())["agent"]["interview-coach"]
    assert agent_cfg["tools"] == {"*": False} and agent_cfg["mode"] == "primary"
    assert msg["model"] == {"providerID": "opencode-go", "modelID": "test-model"}
    assert "Them: hi" in msg["parts"][0]["text"]

    # second question reuses the same OpenCode session (context preserved)
    p.answer("And how do they communicate?", [])
    assert sum(1 for m, pth, _ in mock_opencode.requests if pth.startswith("/session?")) == 1


def test_opencode_grade(mock_opencode, tmp_path):
    p = OpenCodeProvider(mock_opencode.url, directory=str(tmp_path))
    fb = p.grade("q", Answer("a", ["isolation", "shared memory", "cost"]), "they have memory")
    assert fb.missed == ["context switch cost"] and fb.tip


@pytest.mark.parametrize("mode", ["structured", "garbage_once"])
def test_opencode_structured_reply_and_retry_on_non_json(mode, tmp_path):
    m = MockOpenCode(mode=mode)
    try:
        a = OpenCodeProvider(m.url, directory=str(tmp_path)).answer("q?", [])
        assert a.key_points == ["isolation", "shared memory", "context switch cost"]
        assert m.calls == (2 if mode == "garbage_once" else 1)
    finally:
        m.close()


def test_opencode_error_is_reported_readably(tmp_path):
    m = MockOpenCode(mode="expired")
    try:
        with pytest.raises(ProviderError, match="API key has expired"):
            OpenCodeProvider(m.url, directory=str(tmp_path)).answer("q?", [])
    finally:
        m.close()


def test_unreachable_server_health_is_false():
    ok, msg = OpenCodeProvider("http://127.0.0.1:9", timeout=1).health()
    assert not ok and "cannot reach" in msg


def test_openai_compatible_provider(tmp_path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            H.last = body
            content = json.dumps({"short_answer": "s", "key_points": ["a", "b", "c"],
                                  "python_example": ""})
            data = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        p = OpenAICompatibleProvider(f"http://127.0.0.1:{srv.server_address[1]}", "local-model")
        a = p.answer("What is a mutex?", [])
        assert a.key_points == ["a", "b", "c"] and H.last["model"] == "local-model"
        p.answer("And a semaphore?", [])
        assert len(H.last["messages"]) == 4  # system + previous turn (2) + new question
    finally:
        srv.shutdown()


# -- practice flow -------------------------------------------------------------------------

class FakeProvider:
    name = "fake"

    def __init__(self, fail=False, delay=0.0):
        self.fail, self.delay, self.questions = fail, delay, []

    def health(self):
        return True, "fake"

    def answer(self, question, context):
        time.sleep(self.delay)
        self.questions.append((question, context))
        if self.fail:
            raise ProviderError("OpenCode: This API key has expired.")
        return Answer("short", ["p1", "p2", "p3"], "", "fake/model", 0.1)

    def grade(self, question, answer, spoken):
        return Feedback(["p1"], ["p2", "p3"], "tip")

    def close(self):
        pass


def utterance(source, text):
    return {"type": "transcript", "event": {"kind": "utterance", "source": source,
                                            "segment_id": -1, "text": text, "start": 0,
                                            "end": 1}}


def wait_for(pred, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def make_assistant(provider, enabled=True, **kw):
    msgs = []
    a = PracticeAssistant(AssistantConfig(**kw), msgs.append, lambda cfg: provider)
    if enabled:
        a.set_enabled(True)
    return a, msgs


def test_question_from_them_becomes_answered_card_with_spoken_answer_and_feedback():
    prov = FakeProvider()
    a, msgs = make_assistant(prov)
    a.on_message(status("system"))
    a.on_message(utterance("system", "Thanks for joining us today."))   # context, not a question
    a.on_message(utterance("system", "How does Python's GIL affect threads?"))
    assert wait_for(lambda: a.cards and next(iter(a.cards.values()))["status"] == "ready")
    q, ctx = prov.questions[0]
    assert q.startswith("How does") and ctx == ["Them: Thanks for joining us today."]
    a.on_message(utterance("mic", "Only one thread runs Python bytecode at a time."))
    card = next(iter(a.cards.values()))
    assert card["spoken"] == ["Only one thread runs Python bytecode at a time."]
    a.grade(card["id"])
    assert wait_for(lambda: card["feedback_status"] == "ready")
    assert card["feedback"]["missed"] == ["p2", "p3"]
    assert any(m["type"] == "assistant_card" for m in msgs)


def status(source):
    return {"type": "status", "state": "running", "source": source}


def test_mic_only_practice_takes_questions_from_me_and_answers_follow():
    prov = FakeProvider()
    a, _ = make_assistant(prov)
    a.on_message(status("mic"))
    assert a.state()["listen_to"] == ["mic"]
    a.on_message(utterance("mic", "How does Python manage memory?"))
    assert wait_for(lambda: len(prov.questions) == 1)
    a.on_message(utterance("mic", "It uses reference counting plus a cycle collector."))
    card = next(iter(a.cards.values()))
    assert card["question"] == "How does Python manage memory?"
    assert card["spoken"] == ["It uses reference counting plus a cycle collector."]


def test_with_system_audio_mic_utterances_are_not_questions_and_duplicates_skipped():
    prov = FakeProvider()
    a, _ = make_assistant(prov)
    a.on_message(status("both"))
    a.on_message(utterance("mic", "How would you design a rate limiter for a REST API?"))
    a.on_message(utterance("system", "How would you design a cache for a database?"))
    a.on_message(utterance("system", "How would you design a cache for a database"))
    assert wait_for(lambda: len(prov.questions) == 1)
    time.sleep(0.2)
    assert len(a.cards) == 1


def test_disabled_assistant_sends_nothing():
    prov = FakeProvider()
    a, _ = make_assistant(prov, enabled=False)
    a.on_message(status("system"))
    a.on_message(utterance("system", "How does Python's GIL affect threads?"))
    time.sleep(0.2)
    assert not a.cards and not prov.questions


def test_provider_failure_marks_card_without_raising():
    a, _ = make_assistant(FakeProvider(fail=True))
    a.on_message(status("system"))
    a.on_message(utterance("system", "What is a deadlock in concurrency?"))
    assert wait_for(lambda: a.cards and next(iter(a.cards.values()))["status"] == "error")
    assert "expired" in next(iter(a.cards.values()))["error"]


def test_typed_question_and_mode_switch():
    a, msgs = make_assistant(FakeProvider())
    card = a.ask("What is a closure in Python?")
    assert card["source"] == "typed"
    assert wait_for(lambda: a.cards[card["id"]]["status"] == "ready")
    a.set_mode("immediate")
    assert msgs[-1]["type"] == "assistant_state" and msgs[-1]["mode"] == "immediate"
    with pytest.raises(ValueError):
        a.set_mode("hidden")


# -- API ----------------------------------------------------------------------------------

def test_assistant_api_endpoints():
    from fastapi.testclient import TestClient

    from openinterview.config import Config
    from openinterview.web.server import create_app

    prov = FakeProvider()
    cfg = Config()
    cfg.storage.save_sessions = False
    app = create_app(cfg, provider_factory=lambda cfg: prov)
    with TestClient(app) as client:
        assert client.get("/api/assistant").json()["enabled"] is False
        st = client.post("/api/assistant", json={"enabled": True, "mode": "immediate"}).json()
        assert st["enabled"] and st["healthy"] and st["mode"] == "immediate"
        card = client.post("/api/assistant/ask", json={"question": "What is a mutex?"}).json()
        assistant = app.state.assistant
        assert wait_for(lambda: assistant.cards[card["id"]]["status"] == "ready")
        assert client.post(f"/api/assistant/{card['id']}/grade").status_code == 200
        assert client.post("/api/assistant/999/grade").status_code == 404
        assert client.post("/api/assistant/ask", json={"question": " "}).status_code == 400
        with client.websocket_connect("/ws") as ws:
            types = [json.loads(ws.receive_text())["type"] for _ in range(3)]
            assert types == ["snapshot", "metrics", "assistant_snapshot"]


def test_target_is_added_to_coach_prompt():
    from openinterview.assistant.providers import coach_prompt

    p = coach_prompt("", {"company": "Acme", "role": "Backend engineer",
                          "job_description": "We build payment APIs in Python."})
    assert "Company: Acme" in p and "Role: Backend engineer" in p and "payment APIs" in p
    assert "never guess" in p
    assert "Company:" not in coach_prompt("", {})


def test_new_session_clears_cards_and_sets_provider_context():
    class CtxProvider(FakeProvider):
        def __init__(self):
            super().__init__()
            self.contexts = []

        def set_context(self, profile_text, target):
            self.contexts.append(target)

    prov = CtxProvider()
    a, msgs = make_assistant(prov)
    a.on_message(status("system"))
    a.on_message(utterance("system", "How does Python manage memory?"))
    assert wait_for(lambda: len(a.cards) == 1)
    a.on_message({"type": "status", "state": "loading", "source": "system",
                  "target": {"company": "Acme"}})
    assert not a.cards and a.target == {"company": "Acme"}
    assert wait_for(lambda: prov.contexts == [{"company": "Acme"}])
    assert msgs[-1]["type"] == "assistant_snapshot" and msgs[-1]["cards"] == []
    # the same question in the new session is asked again (fresh duplicate memory)
    a.on_message(utterance("system", "How does Python manage memory?"))
    assert wait_for(lambda: len(a.cards) == 1)


def test_mock_interview_script_matches_the_detector(capsys):
    from openinterview import mock

    assert mock.dry_run(mock.script()) == 0
    assert mock.dry_run(mock.script(quick=True)) == 0
    lines = mock.script()
    assert {ln.expect for ln in lines} == {"question", "ignore"}
