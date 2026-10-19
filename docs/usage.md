# Usage

[← README](../README.md) · Command line, the app window, sessions and history, and the web dashboard.

## Command line

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
uv run openinterview -l yue run --source system         # Cantonese interview (Whisper turbo)
```

Models:

| Name | Backend |
|---|---|
| `tiny.en`, `base.en`, `small.en` (default), `large-v3-turbo` | MLX Whisper |
| `parakeet-v2` | NVIDIA Parakeet TDT 0.6B via MLX |
| `apple` | Apple SpeechTranscriber (system model) |

MLX weights download once to the Hugging Face cache. On a TTY, the current partial is
shown in grey: committed words, then `|`, then the still-changing tail.

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
