"""Personal profile: template handling, CV import, prompt injection, API."""

import json
import subprocess

import pytest

from openinterview.assistant import profile as prof
from openinterview.assistant.providers import COACH_PROMPT, OpenCodeProvider, coach_prompt


def test_template_counts_as_empty(tmp_path):
    path = prof.ensure_template(tmp_path / "p.md")
    assert path.read_text() == prof.TEMPLATE
    assert prof.load(path) == ""
    assert prof.info(path)["loaded"] is False


def test_filled_profile_loads_without_comments(tmp_path):
    path = prof.ensure_template(tmp_path / "p.md")
    path.write_text(prof.TEMPLATE.replace("- Name:", "- Name: Alex Chan"))
    text = prof.load(path)
    assert "Alex Chan" in text and "<!--" not in text
    assert prof.info(path)["words"] > 0


def test_long_profile_is_truncated(tmp_path):
    path = tmp_path / "p.md"
    path.write_text("x " * prof.MAX_CHARS)
    assert len(prof.load(path)) == prof.MAX_CHARS
    assert prof.info(path)["truncated"]


def test_import_text_cv_replaces_empty_template(tmp_path):
    dest = prof.ensure_template(tmp_path / "p.md")
    cv = tmp_path / "cv.txt"
    cv.write_text("Alex Chan\nBackend intern, Octopus Cards")
    prof.import_cv(cv, dest)
    text = dest.read_text()
    assert text.startswith("## CV (cv.txt)") and "Octopus Cards" in text


def test_import_keeps_existing_notes(tmp_path):
    dest = tmp_path / "p.md"
    dest.write_text("## Target role\n- Backend engineer at a fintech")
    cv = tmp_path / "cv.md"
    cv.write_text("Alex Chan")
    prof.import_cv(cv, dest)
    text = dest.read_text()
    assert text.index("Target role") < text.index("## CV")


def test_import_word_document_via_textutil(tmp_path):
    src = tmp_path / "cv.txt"
    src.write_text("Alex Chan, BEng Computer Science")
    docx = tmp_path / "cv.docx"
    subprocess.run(["textutil", "-convert", "docx", str(src), "-output", str(docx)], check=True)
    assert "BEng Computer Science" in prof.extract_text(docx)


def test_unsupported_format_rejected(tmp_path):
    bad = tmp_path / "cv.xyz"
    bad.write_text("x")
    with pytest.raises(ValueError, match="unsupported"):
        prof.extract_text(bad)


def test_coach_prompt_includes_profile_and_no_invention_rule():
    assert coach_prompt("") == COACH_PROMPT
    p = coach_prompt("- Name: Alex Chan")
    assert "Alex Chan" in p and "do NOT make it up" in p


def test_opencode_agent_gets_profile_and_new_session_on_change(tmp_path):
    provider = OpenCodeProvider("http://127.0.0.1:9", directory=str(tmp_path), profile="- Name: Alex")
    agent = json.loads((tmp_path / "opencode.json").read_text())["agent"]["interview-coach"]
    assert "Alex" in agent["prompt"]
    provider.session_id = None
    provider.set_profile("- Name: Sam")
    assert provider.directory != str(tmp_path)  # fresh directory: OpenCode caches config
    with open(f"{provider.directory}/opencode.json") as f:
        assert "Sam" in json.load(f)["agent"]["interview-coach"]["prompt"]


def test_profile_api_import_and_state(tmp_path):
    from fastapi.testclient import TestClient

    from openinterview.config import Config
    from openinterview.web.server import create_app

    cfg = Config()
    cfg.assistant.profile_path = str(tmp_path / "profile.md")
    cv = tmp_path / "cv.txt"
    cv.write_text("Alex Chan, intern at Octopus Cards")
    with TestClient(create_app(cfg)) as client:
        assert client.get("/api/profile").json()["loaded"] is False
        assert client.post("/api/profile/edit").json()["exists"] is True
        info = client.post("/api/profile/import", json={"path": str(cv)}).json()
        assert info["loaded"] and info["words"] > 0
        assert client.post("/api/profile/import", json={"path": "/nope.pdf"}).status_code == 400
        assert client.get("/api/assistant").json()["profile"]["loaded"] is True
