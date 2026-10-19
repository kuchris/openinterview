# OpenInterview (macOS, local)

Local, low-latency speech recognition for technical-interview practice and meetings where
recording and AI assistance are permitted. Everything runs on-device (MLX on the Apple
Silicon GPU); no audio leaves the machine, and audio is never written to disk.

See [PLAN.md](PLAN.md) for the full design and phase plan.

**Status: all five phases implemented.** Working so far:

- An optional practice assistant (OpenCode), disabled by default.
- A native macOS app window (`openinterview app`), plus a local web dashboard as an
  alternative.
- Mic and system-audio capture, separately or both at once.
- A BlackHole fallback when the native system-audio helper can't run.
- Automatic recovery when helpers crash, hang, or the output device changes.
- Speaker-echo suppression in "both" mode.
- Silero VAD segmentation, with live partials (LocalAgreement) and finals.
- Hallucination and duplicate filtering, plus sentence assembly.
- Five ASR backends: Whisper ×4, Parakeet, and Apple SpeechAnalyzer.
- A latency/WER/memory benchmark.

## Requirements

- Apple Silicon Mac, macOS 14.2+ (system-audio taps need 14.2; mic works on older versions)
- Xcode or Command Line Tools (for the Swift helpers); the Apple ASR backend needs macOS 26
- `ffmpeg` (only for `transcribe-file` / benchmarks): `brew install ffmpeg`

## Install

```bash
scripts/setup.sh
```

This script:

1. installs dependencies with `uv`;
2. builds `bin/sysaudio` and `bin/appleasr`;
3. downloads `small.en` and the Silero VAD model (2 MB, SHA-256 pinned);
4. runs `openinterview check`.

For Parakeet, also run `uv sync --extra parakeet`.

## Usage

```bash
uv run openinterview app                                 # the app window (recommended)
uv run openinterview serve                               # web dashboard at http://127.0.0.1:8765
uv run openinterview check --probe                       # environment + permission report
uv run openinterview devices                             # list input devices
uv run openinterview run --source mic                    # live mic transcript (Ctrl-C to stop)
uv run openinterview run --source mic --device "MacBook Air"
uv run openinterview run --source system                 # other side of a call
uv run openinterview run --source system --system-backend blackhole
uv run openinterview --model base.en run --source both   # use headphones in this mode
uv run openinterview --model apple run --source system   # Apple on-device streaming ASR
uv run openinterview transcribe-file talk.wav --speed 1 --utterances  # replay a file live
uv run openinterview run --no-audio                      # save the session as text only
uv run openinterview run --no-save                       # save nothing
uv run openinterview mock --app                          # scripted mock interview test
```

Models:

| Name | Backend |
|---|---|
| `tiny.en`, `base.en`, `small.en` (default), `large-v3-turbo` | MLX Whisper |
| `parakeet-v2` | NVIDIA Parakeet TDT 0.6B via MLX |
| `apple` | Apple SpeechTranscriber (system model) |

MLX weights download once to the Hugging Face cache. On a TTY, the current partial is
shown in grey: committed words, then `|`, then the still-changing tail.

## macOS permissions

The two capture paths are attributed differently by macOS:

| Capability | Who gets the permission | Where to enable |
|---|---|---|
| Microphone | **The app that launches Python** (Terminal, iTerm2, VS Code, ...) | System Settings → Privacy & Security → **Microphone** |
| System audio | **`sysaudio` itself** | System Settings → Privacy & Security → **Screen & System Audio Recording** → "System Audio Recording Only" |

**System audio.** On launch, `bin/sysaudio` re-executes itself with TCC "responsibility"
disclaimed, so macOS treats it as its own client. The first `run --source system` shows a
dialog asking whether **sysaudio** may record system audio; click Allow.

Why this is needed: otherwise macOS checks the launching app, which may have no
audio-capture usage string. Some launchers also hand responsibility to an inner bundle,
so enabling the app you *see* does nothing. In both cases the tap is silently denied.

- `sysaudio --inherit-permissions` restores the default behaviour (permission goes to the
  launching app).
- `sysaudio` is ad-hoc signed, so rebuilding it invalidates the grant. macOS asks again on
  the next run, and capture is silent until you click Allow. If no dialog appears, remove
  the old `sysaudio` entry in the settings list and run again.

**Diagnosing an empty transcript.** If capture runs but no text appears:

- `check --probe` reports all-zero mic input → microphone access is denied.
- The `run --source system` summary shows `"nonzero_frames": 0` while audio is playing →
  system-audio access is denied. A denied tap delivers silence, not an error, so macOS
  gives no other signal.

After changing the microphone permission, quit and reopen the terminal app.

## App window

```bash
uv run openinterview app
```

A native SwiftUI window (`bin/openinterview-ui`, built by `scripts/build_helper.sh`). It
starts the backend on a private localhost port and stops it when you close the window.

- **Toolbar:**
  - a recording indicator (red, pulsing while capturing);
  - **Answers**, which turns the assistant on or off (remembered between runs);
  - ⚙︎ settings: source, model and mic;
  - **Start / Stop** (⌘R). Start opens a short **New session** sheet (see below);
  - **Transcript**, which opens the full transcript in a side panel;
  - **History**, which opens past sessions.
- **Sidebar:** the questions, newest first, each with a status icon (thinking / answered /
  error) and who asked it. Below the list is a field to type a question. The view follows
  the newest question unless you have clicked an older one.
- **Main area:** the selected question in large type, then:
  - a short answer you can say out loud;
  - numbered key points;
  - a Python example with a Copy button;
  - **Your answer**: what you said after the question, with "Check my answer" feedback.

  Answers appear automatically (`assistant.mode: immediate`). Set `mode: practice` if you
  want to reveal them by hand in the web dashboard.
- **Bottom:** a two-line live caption bar, with the latest line and the grey partial, final
  latency and OpenCode status. The full transcript stays out of the way in the side panel.

It is a normal, visible window: no overlay, no hiding from screen sharing.

## Sessions and history

Each Start begins a new session. The **New session** sheet asks for:

- a **title**;
- the **company**, **role** and an optional **job description**. These go into the
  assistant's prompt for this session only, so answers ("why do you want to work here?")
  are tailored to that interview. The assistant starts fresh each session.
- **Save this session** and **Record audio** toggles.

The sheet remembers your last entries. **Clear** empties them.

Saved sessions stay on this Mac, one folder each:

```
~/.openinterview/sessions/20261019-160512-acme-backend/
    session.json       title, times, source, model, target, Q&A cards
    transcript.jsonl   finals in order (echo retractions included)
    mic.m4a            audio per source, AAC 32 kbps (~15 MB/hour)
    system.m4a
```

The **History** window (toolbar clock icon) lists them. For each session you can:

- play back the audio;
- read the questions and answers and the transcript;
- show the folder in Finder;
- delete it (removes the folder).

Set the defaults in `config.yaml`:

```yaml
storage:
  save_sessions: true   # false: keep nothing on disk
  save_audio: true      # false: text only
  sessions_dir: null    # default ~/.openinterview/sessions
```

Audio recording captures the other party too. Only record with their consent, and where
it is allowed.

## Mock interview test

`openinterview mock` checks the whole chain: audio, ASR, question detection and
answers. An "interviewer" voice (macOS `say`) reads 14 lines for an "Acme Pay backend
engineer" interview. The lines mix technical and behavioural questions with things that
must be ignored: small talk, statements and a repeated question.

```bash
uv run openinterview mock                 # text only: what the detector decides per line
uv run openinterview mock --app           # with the app open: run a session and report
uv run openinterview mock --app --quick   # 6 lines
uv run openinterview mock --play          # only speak (start a session yourself)
uv run openinterview mock --render mock.wav   # 16 kHz WAV for transcribe-file
```

`--app` (or `--url` for `serve`) does the following:

1. starts a system-audio session with the Acme Pay target, turning on the assistant if
   it is off;
2. speaks each line and waits for its answer;
3. stops and prints, for each line, expected vs detected, the answer time and the start
   of each answer.

It uses an English voice (Samantha, Daniel or Karen) even if your Mac's default voice is
another language. `--voice NAME` picks one. The run is saved as "Mock interview (test)"
in History unless you add `--no-save` (`openinterview --no-save mock --app`).

## Web dashboard (alternative)

```bash
uv run openinterview serve
```

Then open <http://127.0.0.1:8765>. The server binds to localhost only.

- **Controls:** source (mic / system / both), model, mic device, an opt-in "Save
  transcript", and Start/Stop. Controls lock while recording.
- **Visible recording indicator:** a red pulsing "Recording" badge whenever capture is on.
- **Status:** per-source state, and the system backend (tap / blackhole). Reconnects and
  errors show inline; a "degraded" badge appears when partials are paused under load.
- **Transcript:** finals labelled Me / Them with timestamps. The live partial shows below,
  with committed words darker and the changing tail grey. Echo retractions remove lines.
- **Latency panel:** final latency p50/p95, first partial, ASR time per final, RTF, queue
  depth, dropped finals / echo removals, memory. Updates every second.
- **Export .md** downloads the in-memory transcript. "Save session" and "Record audio"
  control what is kept on disk (see [Sessions](#sessions-and-history)).

WebSocket messages (`/ws`): `snapshot`, `status`, `source`, `transcript` (partial / final /
utterance / retract), `metrics`. Partials are coalesced to one per source per 100 ms.

## Practice assistant (optional, OpenCode)

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

   Behavioural questions get a model answer with `[placeholders]` for your own details, and
   key points that outline the structure (e.g. STAR). They get no code.

   Near-duplicates (≥ 85% similar to a recent question) are skipped. At most one request is
   in flight; the small backlog drops its oldest entry.
2. **Practice mode (default):** the card says *"Your turn: answer out loud"*. Your mic
   utterances are attached to the card. Then click **Reveal answer** to see:
   - a short English answer;
   - 3–5 key technical points;
   - an optional Python example.

   **Check my answer** asks the model which key points you covered and which you missed.
   **Immediate mode** shows answers right away (for meeting notes).
3. **Context:** one OpenCode session is kept per run, so follow-ups ("and its time
   complexity?") keep context. Recent transcript lines are also included with each
   question.

You can also type a question in the panel.

### Your profile (personalised behavioural answers)

Without a profile, behavioural answers ("Tell me about yourself", "Why this major?") are
templates with `[placeholders]`. Add your background and they are written about you:

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

- **Facts only.** Answers state only facts from your profile. When it doesn't cover what's
  asked (say, no weakness written down), the answer keeps a `[placeholder]` and a key point
  tells you what to add. Nothing is made up. Verified live: without a weakness in the
  profile, the answer was "My honest weakness is [your weakness]…" with "Add: [your weakness
  + fix] to profile".
- **Stored outside the project** (`assistant.profile_path`, default
  `~/.openinterview/profile.md`), so it never ends up in git.
- **Privacy:** importing sends your CV to your AI provider (OpenCode → the model you chose)
  to organise it. After that, the profile is sent with each question.

### Setup

```bash
mkdir -p ~/opencode-scratch && cd ~/opencode-scratch   # an empty folder
opencode serve --port 4096
```

Then toggle **Practice assistant → On** in the dashboard, or set `assistant.enabled: true`.
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

## System audio backends

| Backend | When | Notes |
|---|---|---|
| **Process tap** (`bin/sysaudio`) | default; macOS 14.2+ and helper built | Captures every app's output. Needs no audio routing changes. |
| **BlackHole** | `auto` fallback, or `--system-backend blackhole` | A virtual loopback device read through PortAudio. Works on older macOS. |

`audio.system_backend: auto` picks the tap. It falls back to BlackHole if the helper is
missing, the OS is too old, or the tap fails to start within 5 s while BlackHole is
installed. `openinterview check` shows which backend will be used.

BlackHole setup (only needed for the fallback):

1. `brew install blackhole-2ch`, or install from <https://existential.audio/blackhole/>.
2. Open **Audio MIDI Setup** → **+** → **Create Multi-Output Device**. Tick your real
   speakers/headphones *and* "BlackHole 2ch". Put the real device first, and enable drift
   correction on BlackHole.
3. Select the Multi-Output Device as the system output, or as the call app's speaker.
4. Note: the volume keys don't control a Multi-Output Device; set volume per device.

The app refuses to use BlackHole as the *mic* device in `both` mode, since that would
transcribe the other side twice.

**Recovery.** The tap helper is supervised:

- **Crash, or no heartbeat for 5 s:** restart with backoff (1 → 15 s).
- **Default output device changes** (AirPods connect, headset unplugged): the helper
  exits with code 75 and is restarted at once on the new device.
- **No app playing:** the tap delivers no frames at all; this is normal. When audio
  resumes, the gap is zero-filled so the system timeline stays aligned with wall-clock time.
- **Mic / BlackHole streams:** a watchdog reopens them after a 2 s stall. PortAudio resets
  are shared and rate-limited, so one source's reconnect doesn't kill the other.
- **`appleasr` exits mid-session:** it is relaunched and continues from the live position.

## Both mode and speaker echo

Mic ("Me") and system ("Them") are separate streams, so there is no digital loopback.
**Use headphones.** Without them, the mic hears the interviewer through the speakers.
Two layers remove that echo (`pipeline.echo_suppression`, on by default, active only in
`both` mode):

1. **Audio gate, before ASR.** For each mic segment, the gate compares the mic's
   20 ms loudness envelope with the system audio's envelope, at lags of −0.1…0.6 s.
   - The segment is echo if the correlation is ≥ 0.6 *and* at most 25% of voiced mic frames
     fall where the system was quiet.
   - When you talk over or between the interviewer's words, you add energy where the
     system is silent. That breaks the second condition, so double-talk is kept.
   - Echo segments are never transcribed, which also saves compute.
2. **Text match, after ASR.** A mic final whose words match (≥ 60%) a time-overlapping
   system final is dropped. If the mic final was already shown, a `retract` event removes
   it.

This is not full acoustic echo cancellation: echo *inside* a genuine double-talk segment
is kept. In the live test (built-in mic, playback audible in the room), the gate removed
one mic segment and the text match dropped another, so no duplicate "Me" lines appeared.

## How it works

```
mic (PortAudio cb) ─┐                         ┌ segmenter thread per source ┐
                    ├► ring buffer (30 s) ───►│ Silero VAD (ONNX, 32 ms)    │─► scheduler ─► inference thread
sysaudio (Swift)  ──┘   per source, bounded   │ onset / silence / 15 s cap  │   finals FIFO    (owns the model)
                                              └─────────────────────────────┘   1 partial/src   │
                       LocalAgreement ◄─ partials ─┬─ filters ◄──────────────────────────────────┘
                       sentence assembly ◄─ finals ┘   (hallucination, repeats, duplicates)
```

- **Capture never blocks.** Callbacks only copy into a fixed-size, timestamped ring buffer.
  Mic and system audio stay in separate buffers end to end ("Me" / "Them").
- **Segmentation.** Silero VAD v6 runs via onnxruntime (no PyTorch). A segment:
  - opens after 160 ms of speech, with 200 ms of pre-roll;
  - closes after 600 ms of silence;
  - is hard-capped at 15 s. Long speech is cut at the least-voiced frame in the last 2 s,
    with no gap and no overlap, so text is never duplicated across cuts.
- **Chunked, not streaming (Whisper / Parakeet).** Each partial re-transcribes the open
  segment: first 0.5 s after onset, then every 1 s. The 15 s cap bounds that cost.
  LocalAgreement-2 commits a word only once two consecutive hypotheses agree, so the stable
  text never flickers or repeats. The final is one more pass over the complete segment.
- **Genuinely streaming (Apple).** `bin/appleasr` feeds SpeechAnalyzer continuously and
  relays its volatile and final results. The model runs in a system XPC service,
  apparently on the Neural Engine, not in this process.
- **Filters.**
  - Whisper's no-speech, log-prob and compression-ratio guards.
  - A blocklist of silence hallucinations ("Thank you for watching"). It is applied only
    when the decode is unconfident or the audio is mostly non-speech, so a real "Thank you."
    survives.
  - Collapsing of n-gram loops ("the the the").
  - Suppression of identical repeated finals of 4+ words; short repeats like "Okay." are kept.
- **Utterances.** Finals without end punctuation merge with the next final from the same
  speaker. Utterances are released on `.?!`, or after a 1.2 s quiet gap. This is the input
  for Phase 5 question detection.
- **Overload policy** (`pipeline.max_finals`, `pipeline.degrade_at`):
  - Finals are never superseded; beyond 8 queued, the oldest is dropped and counted.
  - Each source keeps at most one pending partial; newer ones replace older ones.
  - While 2+ finals are queued, partials are skipped ("degraded"), so live captions pause
    but sentences keep flowing.

## Benchmarks

`bench/benchmark.py` replays a fixed stream through the **real live pipeline** at real-time
speed: 12 LibriSpeech dev-clean utterances (130 s, 303 words) plus 12 interview-style
technical sentences (52 s, 171 words, synthesized with `say -v Samantha`), separated by 1 s
gaps. Each model runs in a fresh process.

```bash
uv run python bench/benchmark.py prepare
uv run python bench/benchmark.py run --models tiny.en base.en small.en large-v3-turbo parakeet-v2 apple
```

Measured on an **Apple M2 (16 GB), macOS 26.6.2**. Raw output is in `bench/results/`.
Vocabulary prompt is on unless noted.

| Model | WER LibriSpeech | WER technical | First partial p50 | Final after speech end p50 / p95 | ASR per final p50 | RTF p50 | Peak RSS | MLX peak |
|---|---|---|---|---|---|---|---|---|
| tiny.en | 10.6% | 4.1% | 584 ms | 716 / 745 ms | 103 ms | 0.026 | 490 MB | 321 MB |
| base.en | 7.6% | 4.1% | 616 ms | 754 / 795 ms | 138 ms | 0.035 | 613 MB | 576 MB |
| **small.en** (default) | 4.6% | **1.8%** | 759 ms | 971 / 1241 ms | 324 ms | 0.081 | 1022 MB | 1172 MB |
| large-v3-turbo | 6.9% | 3.5% | 2390 ms | 2409 / 3263 ms | 1103 ms | 0.266 | 1904 MB | 2041 MB |
| parakeet-v2 | 3.0% | 6.4% | 634 ms | 833 / 1017 ms | 216 ms | 0.045 | 1425 MB | 1976 MB |
| apple | **1.3%** | 11.1% | n/a¹ | 1033 / 1304 ms | n/a | n/a | 254 MB + 124 MB helper² | — |
| base.en, no prompt | 8.6% | 3.5% | 615 ms | 753 / 814 ms | 135 ms | 0.034 | 594 MB | 526 MB |
| small.en, no prompt | 4.3% | 3.5% | 747 ms | 961 / 1214 ms | 313 ms | 0.077 | 985 MB | 1172 MB |

¹ Apple's volatile results carry no per-word timing, so text lag cannot be measured from
them. Observed by hand: volatile text trails speech by about 1 s.
² Plus the system `localspeechrecognition` service (about 33 MB RSS during the run).
Its model memory is not attributable to this process.

What the numbers say:

- **Why small.en is the default.** It is best on technical vocabulary (3 errors in 171
  words) and second-best on read speech, with finals about 1 s after you stop speaking. On
  an M2 it uses about 30% of the GPU for partials.
- **base.en** is the low-resource choice: about half the memory, finals 0.2 s sooner, about
  twice the errors.
- **large-v3-turbo** is too slow on an M2 for 1 s partial updates. Partials were superseded
  under load, and it was *less* accurate than small.en on this set.
- **Parakeet** is excellent on read speech but has no vocabulary prompt, so it misses jargon.
- **Apple SpeechTranscriber** is the most accurate on general speech, nearly free in process
  memory, and leaves the GPU idle. But it mangles jargon ("Kubernates", "redisk cash",
  "item potent", "Asencio"). `AnalysisContext.contextualStrings` had **no measurable
  effect** with SpeechTranscriber in my tests. With `--dictation`, `DictationTranscriber`
  was worse overall.
- **Vocabulary prompt.** The technical-set numbers are within one or two words of each
  other (1 word = 0.6%), so the prompt's effect is small and noisy at this sample size.
- **Final latency.** About 600 ms of the final latency is the deliberate end-of-speech
  silence wait (`pipeline.min_silence_s`). VAD costs about 0.7 ms per 32 ms frame while the
  GPU is busy (about 2% of one core). No run dropped a final.

Caveats: the sample is small, and the technical set is synthetic speech from one voice.
Treat differences of a point or two as noise. Re-run on your machine and voice before
relying on them.

Synthetic test audio: macOS's default `say` voice is badly out-of-distribution for small
Whisper models (tiny.en output was nonsense). `say -v Samantha` transcribes verbatim.

## Tests

```bash
uv run pytest
```

Covers:

- **Audio buffer:** wraparound, overrun accounting, bounded memory, concurrent
  reader/writer.
- **Segmenter:** onset/pre-roll/post-pad, hysteresis, max-segment cuts, no overlap.
- **Deduplication:** LocalAgreement commit rules, n-gram loop collapsing, the hallucination
  blocklist, duplicate finals, and incomplete-sentence merging.
- **Scheduler overload policy.**
- **End-to-end pipeline** with a fake engine and energy VAD: partial→final ordering,
  bounded segments, finals surviving overload.
- **Apple result mapping.**
- **Echo:** the envelope gate (echo detected; independent speech and double-talk kept;
  alignment through ring-buffer timestamps), the text-match drop/retract rules, and
  end-to-end dual-source runs with and without suppression.
- **Dashboard:** the HTTP API, WebSocket snapshot/stream/stop/export, conflict and
  validation errors, late-joining clients, and partial coalescing.
- **Assistant:** question detection (technical and behavioural questions vs small talk, statements, short
  fragments, near-duplicates); the OpenCode request shape against a mock server
  (deny-all permissions, plan agent, json_schema, model id, a reused session, text-JSON
  fallback, an expired-key error); the OpenAI-compatible provider with history; the
  practice flow (context, spoken-answer capture, feedback, disabled = no requests,
  provider failure isolated); and the API endpoints.
- **Profile:** the external-assistant prompt (materials, rules, gaps, format), Markdown
  extraction from a chat reply, paste-back with backup and rejection of non-profile text,
  template detection, comment stripping, truncation, CV import (text,
  Word via textutil, keeping existing notes), prompt injection with the no-invention rule,
  a fresh OpenCode session on profile change, and the API.
- **Sources:** helper crash restart, device-change restart without backoff, hung-helper
  restart, idle-without-frames *not* restarting, idle gap zero-fill, BlackHole fallback
  selection, setup instructions when no backend exists, shared PortAudio reset.
- **Helper frame protocol.**
- **Benchmark WER.**
