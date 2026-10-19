"""Your personal profile, used to personalise behavioural answers.

A plain Markdown file, by default ~/.openinterview/profile.md. It lives outside the project
so it never ends up in git. Fill in the template, or import a CV (PDF, Word, RTF, Pages,
text): the AI organises it into this structure using only facts from the CV, marks gaps
"[to add]" and lists questions to answer. The raw CV text is kept next to it as cv.txt.

Privacy: the profile text is sent to the configured LLM provider with each question.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

MAX_CHARS = 12000  # keep prompts bounded (~3k tokens)

TEMPLATE = """\
# My interview profile

<!-- Used to personalise answers to behavioural questions ("Tell me about yourself",
     "Why this role?", "Describe a time..."). Write facts, short bullets are fine.
     Anything you leave out stays as a [placeholder] in answers: nothing is invented.
     This text is sent to your AI provider with each question. -->

## About me
- Name:
- Current role / status:
- Location:

## Target role
- Role and company I'm interviewing for:
- Why I want it:

## Experience
-

## Projects
-

## Education
-

## Skills
-

## Strengths and weaknesses
- Strength:
- Weakness (and what I'm doing about it):

## Stories (STAR: situation, task, action, result)
- Conflict in a team:
- A failure or mistake:
- Something I'm proud of:
"""

TEXTUTIL_TYPES = {".docx", ".doc", ".rtf", ".rtfd", ".pages", ".odt", ".html", ".htm", ".webarchive"}


def default_path() -> Path:
    return Path.home() / ".openinterview" / "profile.md"


def resolve(path: str | None) -> Path:
    return Path(path).expanduser() if path else default_path()


def ensure_template(path: Path) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(TEMPLATE)
    return path


def is_template_only(text: str) -> bool:
    """True if the profile has no real content beyond the template's labels."""
    def strip(t: str) -> set[str]:
        return {ln.strip() for ln in t.splitlines() if ln.strip()}
    return strip(text) <= strip(TEMPLATE)


def load(path: Path) -> str:
    """Profile text for the prompt ('' if missing or still the empty template)."""
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    if is_template_only(text):
        return ""
    # drop HTML comments (template instructions)
    import re

    text = re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()
    return text[:MAX_CHARS]


def extract_text(source: Path) -> str:
    """Extract plain text from a CV file."""
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(source))
        return "\n".join((page.extract_text() or "") for page in reader.pages).strip()
    if suffix in TEXTUTIL_TYPES:
        out = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(source)],
                             capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise ValueError(f"could not read {source.name}: {out.stderr.strip()}")
        return out.stdout.strip()
    if suffix in {".txt", ".md", ".markdown", ""}:
        return source.read_text(errors="replace").strip()
    raise ValueError(f"unsupported CV format: {suffix} (use PDF, Word, RTF, Pages or text)")


def import_cv(source: Path, dest: Path) -> int:
    """Write the CV's text into the profile (keeping any notes already there). Returns chars."""
    text = extract_text(source)
    if not text:
        raise ValueError(f"no text found in {source.name} (is it a scanned image?)")
    existing = dest.read_text(errors="replace") if dest.exists() else ""
    notes = "" if not existing or is_template_only(existing) else existing.strip() + "\n\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"{notes}## CV ({source.name})\n\n{text}\n")
    return len(text)


def info(path: Path) -> dict:
    text = load(path)
    return {"path": str(path), "exists": path.exists(), "loaded": bool(text),
            "words": len(text.split()), "truncated": path.exists()
            and len(path.read_text(errors="replace")) > MAX_CHARS}


def save_cv_text(text: str, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "cv.txt"
    path.write_text(text + "\n")
    return path


def write_generated(dest: Path, markdown: str, to_add: list[str], source_name: str) -> None:
    """Write an AI-organised profile. Notes for the user live in HTML comments, which
    load() strips, so they are never sent to the model."""
    from datetime import date

    header = (f"<!-- Organised from {source_name} by AI on {date.today()}. Check every line: "
              "fix anything wrong and fill in the [to add] parts. This text is sent to your "
              "AI provider with each question. -->\n\n")
    todo = ""
    if to_add:
        todo = ("\n\n<!-- To complete your profile, answer these (write the answers in the "
                "sections above):\n" + "\n".join(f"- {q}" for q in to_add) + "\n-->")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(header + markdown.strip() + todo + "\n")
