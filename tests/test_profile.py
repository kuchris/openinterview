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


class FakeBuilder:
    """Stands in for the LLM: records what it was given, returns a structured profile."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def build_profile(self, cv_text, existing=""):
        self.calls.append((cv_text, existing))
        if self.fail:
            raise RuntimeError("OpenCode unreachable")
        return ("## About me\n- Name: Alex Chan\n\n## Weaknesses\n- [to add]",
                ["Which role are you targeting?", "What is a real weakness?"])

    def health(self):
        return True, "fake"

    def close(self):
        pass


def make_client(tmp_path, builder):
    from fastapi.testclient import TestClient

    from openinterview.config import Config
    from openinterview.web.server import create_app

    cfg = Config()
    cfg.assistant.profile_path = str(tmp_path / "profile.md")
    return TestClient(create_app(cfg, provider_factory=lambda c: builder))


def test_profile_api_ai_import_organises_cv_and_lists_gaps(tmp_path):
    cv = tmp_path / "my_cv.txt"
    cv.write_text("Alex Chan, intern at Octopus Cards")
    builder = FakeBuilder()
    with make_client(tmp_path, builder) as client:
        assert client.get("/api/profile").json()["loaded"] is False
        assert client.post("/api/profile/edit").json()["exists"] is True
        r = client.post("/api/profile/import", json={"path": str(cv)}).json()
        assert r["method"] == "ai" and r["loaded"] and r["to_add"]
        assert builder.calls[0][0] == "Alex Chan, intern at Octopus Cards"
        saved = (tmp_path / "profile.md").read_text()
        assert "## About me" in saved and "Which role are you targeting?" in saved
        # gap questions live in a comment: never part of the prompt text
        assert "Which role" not in prof.load(tmp_path / "profile.md")
        assert (tmp_path / "cv.txt").read_text().startswith("Alex Chan")  # raw text kept
        assert client.post("/api/profile/import", json={"path": "/nope.pdf"}).status_code == 400
        assert client.get("/api/assistant").json()["profile"]["loaded"] is True


def test_profile_import_falls_back_to_raw_text_when_ai_fails(tmp_path):
    cv = tmp_path / "resume.md"
    cv.write_text("Alex Chan, intern at Octopus Cards")
    with make_client(tmp_path, FakeBuilder(fail=True)) as client:
        r = client.post("/api/profile/import", json={"path": str(cv)}).json()
        assert r["method"] == "raw" and "unreachable" in r["warning"] and r["loaded"]
        assert "## CV (resume.md)" in (tmp_path / "profile.md").read_text()


def test_existing_notes_are_passed_to_the_builder(tmp_path):
    from openinterview.assistant.practice import import_profile
    from openinterview.config import AssistantConfig

    dest = tmp_path / "profile.md"
    dest.write_text("## Target role\n- Backend engineer at a fintech")
    cv = tmp_path / "cv.txt"
    cv.write_text("Alex Chan")
    builder = FakeBuilder()
    import_profile(AssistantConfig(profile_path=str(dest)), str(cv), builder)
    assert "Backend engineer at a fintech" in builder.calls[0][1]


def test_glued_words_detects_lost_spaces():
    assert prof.glued_words("Software Engineering Intern at Octopus") == 0
    assert prof.glued_words("SoftwareEngineeringInternatOctopus") == 1


def test_strip_raw_cv_keeps_notes_only():
    text = "## Target role\n- Backend\n\n## CV (cv.pdf)\n\nSoftwareEngineeringIntern"
    assert prof.strip_raw_cv(text) == "## Target role\n- Backend"
    assert prof.strip_raw_cv("## CV (cv.pdf)\nraw") == ""


def test_raw_reimport_replaces_previous_cv_text(tmp_path):
    dest = tmp_path / "profile.md"
    dest.write_text("## Target role\n- Backend\n\n## CV (old.pdf)\n\nOLD TEXT")
    cv = tmp_path / "new.txt"
    cv.write_text("NEW TEXT")
    prof.import_cv(cv, dest)
    text = dest.read_text()
    assert "OLD TEXT" not in text and "NEW TEXT" in text and "Target role" in text


def test_pdf_uses_pdfkit_helper_when_available(tmp_path):
    from openinterview.config import PROJECT_ROOT

    if not (PROJECT_ROOT / "bin" / "pdftext").is_file():
        pytest.skip("bin/pdftext not built")
    src = tmp_path / "cv.txt"
    src.write_text("Software Engineering Intern at Octopus Cards")
    pdf = tmp_path / "cv.pdf"
    subprocess.run(["cupsfilter", str(src)], stdout=pdf.open("wb"), stderr=subprocess.DEVNULL,
                   check=True)
    text = prof.extract_pdf(pdf)
    assert "Software Engineering Intern" in text and prof.glued_words(text) == 0
