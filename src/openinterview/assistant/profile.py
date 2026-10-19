"""Your personal profile, used to personalise behavioural answers.

A plain Markdown file, by default ~/.openinterview/profile.md. It lives outside the project
so it never ends up in git. Fill in the template, or import a CV (PDF, Word, RTF, Pages,
text): the AI organises it into this structure using only facts from the CV, marks gaps
"[to add]" and lists questions to answer. The raw CV text is kept next to it as cv.txt.

Privacy: the profile text is sent to the configured LLM provider with each question.
"""

from __future__ import annotations

import re
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
    # drop HTML comments (template instructions, AI notes)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S).strip()
    return text[:MAX_CHARS]


def extract_text(source: Path) -> str:
    """Extract plain text from a CV file."""
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(source)
    if suffix in TEXTUTIL_TYPES:
        out = subprocess.run(["textutil", "-convert", "txt", "-stdout", str(source)],
                             capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            raise ValueError(f"could not read {source.name}: {out.stderr.strip()}")
        return out.stdout.strip()
    if suffix in {".txt", ".md", ".markdown", ""}:
        return source.read_text(errors="replace").strip()
    raise ValueError(f"unsupported CV format: {suffix} (use PDF, Word, RTF, Pages or text)")


def glued_words(text: str) -> int:
    """Words of 20+ letters: a sign the extractor lost the spaces between words."""
    return sum(1 for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 20)


def extract_pdf(source: Path) -> str:
    """PDF text via Apple's PDFKit (bin/pdftext), falling back to pypdf. Some PDFs' font
    encodings make pypdf glue words together ("SoftwareEngineeringIntern"); PDFKit keeps
    the spacing, so the candidate with fewer glued words wins."""
    from ..config import PROJECT_ROOT

    candidates = []
    helper = PROJECT_ROOT / "bin" / "pdftext"
    if helper.is_file():
        out = subprocess.run([str(helper), str(source)], capture_output=True, text=True, timeout=60)
        if out.returncode == 0 and out.stdout.strip():
            candidates.append(out.stdout.strip())
    if not candidates or glued_words(candidates[0]) > 0:
        from pypdf import PdfReader

        reader = PdfReader(str(source))
        text = "\n".join((page.extract_text() or "") for page in reader.pages).strip()
        if text:
            candidates.append(text)
    if not candidates:
        return ""
    return min(candidates, key=glued_words)


def strip_raw_cv(text: str) -> str:
    """Drop a raw '## CV (file)' section left by an earlier plain import."""
    return re.split(r"(?m)^## CV \(", text, maxsplit=1)[0].strip()


def import_cv(source: Path, dest: Path) -> int:
    """Write the CV's text into the profile (keeping any notes already there). Returns chars."""
    text = extract_text(source)
    if not text:
        raise ValueError(f"no text found in {source.name} (is it a scanned image?)")
    existing = strip_raw_cv(dest.read_text(errors="replace")) if dest.exists() else ""
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


NOTES_HEADING = "## Added while practising"


def add_note(path: Path, topic: str, text: str) -> None:
    """Append a fact the user typed (e.g. a real conflict story) to the profile, under its
    own section, so the next answers can use it. Keeps the AI's trailing comment last."""
    topic, text = " ".join(topic.split()), " ".join(text.split())
    if not text:
        raise ValueError("empty note")
    entry = f"- {topic.rstrip('.')}: {text}" if topic else f"- {text}"
    body = path.read_text(errors="replace").rstrip("\n") if path.exists() else ""
    if is_template_only(body):
        body = ""
    tail = ""
    m = re.search(r"\n*<!-- To complete your profile.*-->\s*$", body, flags=re.S)
    if m:
        body, tail = body[:m.start()], "\n\n" + body[m.start():].strip()
    if NOTES_HEADING in body:
        head, rest = body.split(NOTES_HEADING, 1)
        nxt = re.search(r"\n## ", rest)
        cut = nxt.start() if nxt else len(rest)
        body = head + NOTES_HEADING + rest[:cut].rstrip("\n") + "\n" + entry + rest[cut:]
    else:
        body = (body + "\n\n" if body else "") + NOTES_HEADING + "\n" + entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + tail + "\n")


# -- working on the profile with an external assistant (ChatGPT, Claude, ...) -------------

SECTIONS = ["About me", "Target role", "Experience", "Projects", "Education", "Skills",
            "Strengths", "Weaknesses", "Stories (STAR)"]


def open_questions(path: Path) -> list[str]:
    """The 'To complete your profile' questions left by the AI import (in a comment)."""
    if not path.exists():
        return []
    m = re.search(r"<!-- To complete your profile.*?:\n(.*?)-->", path.read_text(errors="replace"),
                  flags=re.S)
    if not m:
        return []
    return [ln[2:].strip() for ln in m.group(1).splitlines() if ln.startswith("- ")]


def external_prompt(path: Path) -> str:
    """A self-contained prompt to paste into a strong chat model. It interviews you to fill
    the gaps, then returns the finished profile in this app's format."""
    current = load(path) or "(empty)"
    cv_file = path.parent / "cv.txt"
    cv = cv_file.read_text(errors="replace").strip() if cv_file.exists() else "(no CV imported)"
    questions = open_questions(path)
    gaps = "\n".join(f"- {q}" for q in questions) or "- (none listed: find the gaps yourself)"
    sections = "\n".join(f"## {s}" for s in SECTIONS)
    return f"""You are an expert interview coach. Help me finish my interview-preparation \
profile, a document an AI assistant uses to answer interview questions in my voice.

# How to work with me
1. Read my CV and my current profile below. Then list briefly what is missing or weak \
(target role, weaknesses, STAR stories without a clear task or result, vague claims).
2. Interview me to fill those gaps: ask ONE question at a time and wait for my answer. \
Ask follow-ups until a story has a concrete Situation, Task, Action and Result \
(numbers if I have them).
3. When I say "done" (or every gap is filled), output the final profile.

# Rules
- Facts only: use only what is in my CV, my current profile, or what I tell you. Never \
invent employers, dates, numbers, motives or outcomes. If something is still unknown, \
write [to add].
- Keep my numbers and specifics. Write in first person, plain English, short bullets.
- Weaknesses must be real and come with what I'm doing about them (no humble-brags).
- Aim for 3-5 STAR stories that cover: a conflict or disagreement, a mistake or failure, \
leadership or initiative, a technical challenge, something I'm proud of.
- At most 900 words in total.

# Output format (final answer only)
Return the whole profile as ONE Markdown code block (```markdown ... ```), with exactly \
these sections in this order and nothing outside the code block:
{sections}

# Questions already noted as gaps
{gaps}

# My current profile
<<<
{current}
>>>

# My CV (extracted text)
<<<
{cv}
>>>
"""


def extract_markdown(text: str) -> str:
    """Take the profile out of a chat reply: the (last) ```markdown block if present."""
    blocks = re.findall(r"```(?:markdown|md)?[ \t]*\n(.*?)```", text, flags=re.S)
    body = max(blocks, key=len) if blocks else text
    return body.strip()


def paste_profile(text: str, dest: Path) -> dict:
    """Replace the profile with a pasted one (backing up the old file)."""
    from datetime import date

    body = extract_markdown(text)
    headings = re.findall(r"(?m)^## .+", body)
    if len(headings) < 3:
        raise ValueError("that doesn't look like a profile (expected Markdown with ## sections). "
                         "Copy the model's final Markdown block.")
    backup = None
    if dest.exists():
        backup = dest.with_name("profile.backup.md")
        backup.write_text(dest.read_text(errors="replace"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(f"<!-- Pasted on {date.today()} from an external assistant. Check every "
                    "line; this text is sent to your AI provider with each question. -->\n\n"
                    + body + "\n")
    missing = [s for s in SECTIONS if f"## {s}" not in body]
    return {"sections": len(headings), "missing_sections": missing,
            "to_add": body.count("[to add]"), "backup": str(backup) if backup else None}
