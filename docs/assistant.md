# Practice assistant

[← README](../README.md) · AI answers via OpenCode, your profile / CV, and how answers stay factual.

The assistant is disabled by default, and transcription never depends on it. When on, it
turns the interviewer's complete questions into practice cards:

1. A **complete utterance** becomes a card if it looks like an interview question, technical
   or behavioural. By
   default (`listen_to: [auto]`), questions come from "Them" when system audio is captured.
   In **mic-only** mode they come from "Me", so you can practise by reading questions aloud
   and then answering. Utterances are merged finals; partials and fragments are never sent. A
   question needs:
   - a question form (`?`, "how would you…", "explain…", "walk me through…");
   - at least 4 words;
   - an interview cue: a technical term, or a behavioural one ("tell me about yourself",
     "why did you choose…", "describe a time…", strengths, experience, team…). Small talk
     ("How are you?", "Can you hear me?") is ignored (`assistant.filter_small_talk`).

   Behavioural questions get a first-person model answer built from your profile, and key
   points that outline the structure (e.g. STAR). They get no code, and never blanks to fill
   in (see "Facts only" below).

   Near-duplicates (≥ 85% similar to a recent question) are skipped. At most one request is
   in flight; the small backlog drops its oldest entry.
2. **Immediate mode (default):** the answer appears as soon as it is ready:
   - a short answer to say out loud (in the interview's language);
   - 3–5 key points;
   - an optional Python example;
   - a 💡 hint when your profile lacks a fact (see "Facts only" below).

   Your mic utterances after the question are attached to the card, and **Check my
   answer** asks the model which key points you covered and which you missed.
   **Practice mode** (`assistant.mode: practice`, web dashboard) hides the answer until
   you click **Reveal answer**, so you can try first.
3. **Context:** one OpenCode session is kept per run, so follow-ups ("and its time
   complexity?") keep context. Recent transcript lines are also included with each
   question.

You can also type a question in the panel.

## Your profile (personalised behavioural answers)

Without a profile, behavioural answers ("Tell me about yourself", "Why this major?") can
only be general. Add your background and they are written about you:

```bash
uv run openinterview profile import ~/Documents/cv.pdf   # PDF, Word, RTF, Pages or text
uv run openinterview profile edit                        # opens ~/.openinterview/profile.md
```

In the app, the same is under ⚙︎ → **Your profile** (Import CV…, Edit…, Reload).

**Import analyses the CV with the AI.** It doesn't just paste text in:

1. The text is extracted (PDF, Word, RTF, Pages or text). The raw text is kept as
   `~/.openinterview/cv.txt`.
   - PDFs go through Apple's PDFKit (`bin/pdftext`), with pypdf as a fallback. Some PDF
     fonts make pypdf drop the spaces between words ("SoftwareEngineeringIntern"); the
     result with fewer glued words wins.
2. A separate `profile-builder` agent organises it into the profile sections: about me,
   target role, experience (keeping the CV's numbers), projects, education, skills,
   strengths backed by evidence, weaknesses, and 2–3 STAR story drafts built from your
   achievements.
3. It uses only facts from the CV and your existing notes, which are kept. Anything the CV
   doesn't state is written `[to add]` (target role, weaknesses, the *task* in a story…).
4. You get 3–6 questions to answer to complete the profile. They are stored as a comment
   in the file and are never sent with questions.
5. If the AI is unreachable, the raw CV text is imported instead, and you're told why.
   Analysis gets a 180 s timeout, since a long CV can take over a minute. Re-importing
   replaces earlier raw CV text instead of piling it up.

Live, on a fictional CV, it took 11 s. Every fact matched the CV, gaps were marked
`[to add]`, and the questions included "What is a real weakness, and how are you working
on it?". Check the result with **Edit…** and fill in the gaps: the more real stories you
add, the better behavioural answers get.

**Finish it with ChatGPT or Claude.** ⚙︎ → Your profile → **Copy prompt** (or
`openinterview profile prompt`) copies a self-contained prompt with your CV text, your
current profile, the open questions, and strict rules.

The model interviews you one question at a time until every STAR story has a concrete
situation, task, action and result. It may use only facts you give it, and writes
`[to add]` for anything still unknown. When you say "done", it returns the finished profile
as one Markdown block in this app's section format.

Copy that block and click **Paste result** (or `openinterview profile paste`). The previous
version is kept as `profile.backup.md`. Pasted text without `##` sections is rejected, so a
stray chat message can't overwrite your profile. Note that the prompt contains your CV:
paste it only into a service you trust with it.

- **Facts only, and always sayable.** Answers state only facts from your profile and never
  contain `[placeholders]` to fill in mid-interview. When the profile lacks what's asked
  (say, no conflict story), the answer is still something you can say honestly: it uses the
  closest real facts and how you approach such situations, without inventing events,
  employers or numbers. Any leftover `[your …]` / `[Add …]` text from the model is removed.
- **💡 Gap hint + Add to profile.** When an answer had to work around missing facts, a
  hint below it names what's missing ("No real conflict story in your profile").
  **Add to profile…** lets you type the real story in a sentence or two. It's appended
  under `## Added while practising` in your profile, and that question is answered again
  using it. Next time the gap is gone.
- **Check before the interview.** In the New session sheet, **Check my profile for this
  interview** compares your profile with the company, role and job description. It lists
  the 3–5 likely questions your profile can't answer well yet (e.g. "This role uses Go and
  Kafka. What hands-on experience do you have?"), each with what's missing and what to
  write. Answer the ones you can, then **Save answers to profile**. It runs in its own
  throwaway session (agent `interview-prep`) and took ~12 s live.
- **Stored outside the project** (`assistant.profile_path`, default
  `~/.openinterview/profile.md`), so it never ends up in git.
- **Privacy:** importing sends your CV to your AI provider (OpenCode → the model you chose)
  to organise it. After that, the profile is sent with each question.

## Setup

```bash
mkdir -p ~/opencode-scratch && cd ~/opencode-scratch   # an empty folder
opencode serve --port 4096
```

Then turn on **Answers** (✦ in the app's toolbar; remembered between runs), or set
`assistant.enabled: true`.
`openinterview check` shows whether the server is reachable.

How it talks to OpenCode: the documented server API, verified against OpenCode 1.19.3's
`/doc` OpenAPI spec.

- `POST /session?directory=<empty temp dir>` creates the session, with
  `permission: [{"permission": "*", "pattern": "*", "action": "deny"}]`.
- **A dedicated `interview-coach` agent.** The provider writes it into `opencode.json` in
  its own empty working directory: a short coaching prompt, all tools off, temperature 0.3.
  OpenCode's built-in `plan` agent carried about 6.5k tokens of coding-agent instructions,
  which made answers slow and verbose. The coach agent sends about 300 tokens per question.
- `POST /session/{id}/message` sends each question, with `agent: "interview-coach"` and the
  answer rules in `system`:
  - a short answer of 1–2 sentences, at most 45 words;
  - 3–5 key-point cues of at most 8 words each;
  - a Python example of at most 10 lines, only when it helps.

  The reply's text is parsed as JSON, with one retry if it isn't valid JSON.
- OpenCode's own `format: json_schema` is **not** used. OpenCode implements it as a hidden
  `StructuredOutput` tool, which the deny-all permission blocks. DeepSeek V4.1 Flash then
  fails with "Model did not produce structured output".
- **Safety:** tools are disabled on the agent, every tool permission is denied on the
  session, and the working directory is empty. The model can answer, but cannot read or change files or run commands.
- **Model:** `assistant.model` picks one as `"providerID/modelID"` (list them with
  `opencode models`). `null` uses OpenCode's default. Tested with
  `opencode-go/deepseek-v4.1-flash`: 2–4 s per answer. `opencode-go/deepseek-v4-pro` gave
  similar answers but took 12–25 s.
- **Errors** (server down, expired key, timeout) show on the card; transcription keeps
  running. Verified live: with an expired OpenCode Go key, the card showed OpenCode's
  "This API key has expired…" message.

Alternative: `assistant.provider: openai` with `base_url` pointing at any OpenAI-compatible
`/v1` server (for example a local model server). The key is read from the environment
variable named in `assistant.api_key_env`, never from the config file.

Scope: this is a practice tool. No hidden overlays, no concealed recording, no
screen-share evasion. The recording indicator is always visible.
