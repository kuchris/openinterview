# OpenInterview — Local Real-Time ASR Assistant for macOS (Revised Plan)

## Purpose and scope

A low-latency, fully local speech recognition app for macOS (Apple Silicon), used for
**technical interview practice, mock interviews, and meetings where recording and AI
assistance are permitted**.

Out of scope, permanently:

- Hidden or click-through overlays, windows excluded from screen sharing, or any mechanism
  meant to conceal the app or bypass interview / proctoring rules.
- Covert recording. The UI always shows a visible "recording" indicator while capturing.

Target environment: Apple Silicon, macOS 14.2+ (developed on macOS 26), Python 3.11+.

## 1. Technology stack

| Concern | Choice | Notes |
|---|---|---|
| ASR (default) | `mlx-whisper` with `mlx-community/*-mlx` weights | Chunked transcription, **not** native streaming |
| ASR (candidate) | `parakeet-mlx` (NVIDIA Parakeet TDT 0.6B) | Benchmarked against Whisper; often faster and more accurate for English |
| VAD | Silero VAD via **ONNX** (`onnxruntime`) | Avoids pulling in PyTorch (~hundreds of MB) |
| Microphone | `sounddevice` | |
| System audio (primary) | **Core Audio Process Tap** (macOS 14.2+) via a small Swift helper | Audio-only, narrower permission than screen recording |
| System audio (fallback 1) | ScreenCaptureKit audio via the same Swift helper | macOS 13+ |
| System audio (fallback 2) | BlackHole virtual device read via `sounddevice` | No native code required |
| Processing | asyncio + one dedicated inference thread + bounded queues | MLX inference stays on a single thread |
| UI | FastAPI + WebSockets + static HTML/JS | No frontend build step |
| Answers (optional) | `AnswerProvider` interface; OpenCode server is the first implementation | Disabled by default |

### Swift helper for system audio

Using pyobjc directly for Core Audio taps or SCStream callbacks is fragile. Instead, ship
`helpers/sysaudio/` (Swift Package):

- Captures system audio (Process Tap → SCK fallback).
- Writes raw 16 kHz mono float32 PCM frames to stdout, each with a small header
  (monotonic timestamp in ns + sample count).
- Reports errors and permission state as JSON lines on stderr.
- Python (`audio/system_audio.py`) spawns it as a subprocess and reads frames.

If building the helper is not possible (no Xcode or Command Line Tools), the app falls back
to BlackHole automatically and says so in the dashboard.

## 2. Audio capture

Modes: `mic`, `system`, `both`.

- Each source has its own capture → its own ring buffer → its own VAD/segmenter. They are
  only combined at transcript level, which gives source-based speaker labels for free:
  `mic → "Me"`, `system → "Them"`.
- All buffers are timestamped (monotonic clock at capture time).
- Resampling to 16 kHz mono happens once, as close to capture as possible (in the Swift
  helper for system audio; `soxr` or a polyphase resampler for the mic if the device can't
  open at 16 kHz natively).
- The capture callback only copies into a preallocated ring buffer. It never allocates
  large objects, takes long locks, or touches the model.
- Device disconnect: the stream error is caught, the source goes to a `reconnecting` state,
  is retried with backoff, and the status is shown in the dashboard.

### Loopback and echo

Separate streams avoid **digital** loopback only. If the user plays audio through
speakers, the mic picks it up **acoustically** and it gets transcribed twice. Mitigations:

1. Recommended: wear headphones (documented and shown as a hint in `both` mode).
2. Optional: enable macOS Voice Processing IO (AEC) for the mic path through the Swift helper.
3. Cheap safety net: drop a mic segment if its text closely matches a system segment that
   overlaps it in time.

### Permissions

- macOS grants microphone, screen-recording, and system-audio permissions to the
  **app that launches Python** (Terminal, iTerm2, VS Code, etc.), not to Python itself.
  The README states this explicitly.
- On startup, `main.py --check` reports: architecture, macOS version, mic permission status,
  system-audio helper availability, BlackHole presence, and model cache status.

## 3. Local ASR

### Models to benchmark

| Model | Role |
|---|---|
| `whisper-tiny.en` | Fastest baseline |
| `whisper-base.en` | Default starting point |
| `whisper-small.en` | Higher accuracy |
| `whisper-large-v3-turbo` | Accuracy reference, still fast on M-series |
| `parakeet-tdt-0.6b` | Strong English candidate |

The final default is chosen from **measured** latency and WER, not assumed.

### Segmentation and partials

- Silero VAD runs on fixed 512-sample frames (32 ms at 16 kHz).
- A speech segment opens on VAD onset (with ~200 ms pre-roll) and closes after a
  configurable silence (default 600 ms).
- **Maximum window is 15 s.** Long speech is force-cut at the lowest-energy point near the
  limit. This prevents O(n²) cost from re-transcribing an ever-growing buffer.
- While speech is ongoing, the current window is re-transcribed every ~1 s to produce
  partials.
- **LocalAgreement-2** (as in ÚFAL `whisper_streaming`): a word prefix is committed only
  once two consecutive hypotheses agree on it. Committed text is never re-emitted, which
  handles both duplicate suppression and stable partials.
- When the segment closes, the remainder is finalized.
- Incomplete sentences: a segment that ends without terminal punctuation and is followed
  quickly by more speech is merged into the next segment before the sentence is emitted as
  final.

### Hallucination guards

- No inference on windows the VAD marks as non-speech.
- Discard segments where `no_speech_prob` is high, the compression ratio exceeds its
  threshold, or the average log-probability is too low.
- `condition_on_previous_text=False`. A short committed-text prompt is passed instead.
- A blocklist of well-known silence hallucinations ("Thank you for watching", etc.) is
  applied only to very short, low-confidence outputs.

Language is fixed to English. Every word and segment carries a timestamp.

## 4. Latency, overload policy, and benchmarks

### Overload policy (documented in the README)

- Each source → VAD queue is bounded. When it is full, the **oldest audio frame is dropped**
  and a counter is incremented. Capture never blocks.
- Inference queue holds at most **1 partial job per source** (newer replaces older) plus a
  bounded FIFO of final jobs.
- When backlog exceeds a threshold, partial updates are skipped and only finals are
  processed. The dashboard shows a "degraded" badge.

### Metrics (live in the dashboard and in the benchmark script)

- Capture latency: hardware/helper timestamp → enqueue.
- VAD latency: frame enqueue → segmentation decision.
- ASR inference latency per job.
- End-to-end: end of speech → final text on screen.
- Time to first partial.
- Real-time factor (RTF).
- Peak RAM: process RSS **plus** `mx.get_peak_memory()` (MLX uses unified memory, so RSS
  alone under-reports).
- Dropped-frame and skipped-partial counters.

### Benchmark script (`bench/benchmark.py`)

- Replays reference audio files at real-time speed through the full pipeline (no mic needed).
- Dataset: a few LibriSpeech test-clean samples plus a small set of technical-vocabulary
  recordings (e.g. "idempotent", "Kubernetes", "B-tree", "O(n log n)") with reference text.
- Reports latency percentiles (p50/p95), RTF, peak memory, and **WER** per model.
- Results go to `bench/results/<date>.md`. The README only quotes measured numbers.

## 5. Dashboard

Single page served at `http://127.0.0.1:8765` (bound to localhost only).

- Source selector (mic / system / both) and input device picker.
- Model selector (reloads the model on the inference thread).
- Start / Stop, with a clearly visible recording indicator.
- Live transcript: finalized lines labelled "Me" / "Them" with timestamps; current partial
  in grey.
- Metrics strip: ASR latency, end-to-end latency, RTF, queue depth, dropped frames.
- Source and permission status (including fallback-to-BlackHole notices).

Privacy defaults:

- Audio is **never** written to disk.
- Transcript persistence is off by default. When enabled, it writes JSONL/Markdown to
  `transcripts/`.

## 6. Practice assistant (optional, disabled by default)

### Question detection

- Only runs on **finalized** text from the `system` source ("Them").
- Turn end = system source silent for ≥ 1.5 s (configurable). Debounced, so one turn
  produces at most one request.
- A turn is classified as a question with heuristics (question mark, interrogative opener,
  "explain / walk me through / how would you / what is the difference…"), optionally
  confirmed by the LLM.
- Duplicate guard: normalized-text hash plus a similarity check against recent questions.

### Practice flow (default)

1. Question detected → shown as a card: "Your turn — answer out loud."
2. The user answers. Their speech comes in on the `mic` source and is tied to the card.
3. The user clicks **Reveal** (or a configurable delay passes) → the reference answer is
   shown:
   - A short English answer.
   - 3–5 key technical points.
   - An optional Python example.
4. **Feedback**: which key points the user's spoken answer covered and which it missed.

A "show answer immediately" mode exists for meeting note-taking. Practice mode is the
default.

### Provider interface

```python
class AnswerProvider(Protocol):
    async def answer(self, question: str, context: list[Turn]) -> Answer: ...
    async def grade(self, question: str, reference: Answer, user_answer: str) -> Feedback: ...
```

- `OpenCodeProvider`: talks to a locally running `opencode serve` using its documented
  session/message HTTP API. Reuses one session per practice run, which preserves context.
  The exact endpoints are verified against the installed OpenCode version in Phase 5,
  not assumed.
- `OpenAICompatibleProvider`: any OpenAI-compatible endpoint (e.g. a local server).
  This is a lighter-weight alternative, since OpenCode is an agentic coding tool and
  heavier than needed for short Q&A.
- If no provider is reachable, the ASR pipeline is unaffected and the panel shows "assistant
  unavailable".

## 7. Project structure

```
openinterview/
├── README.md
├── PLAN.md
├── pyproject.toml
├── config.yaml
├── helpers/
│   └── sysaudio/              # Swift package: Process Tap / SCK → PCM on stdout
├── src/openinterview/
│   ├── main.py                # CLI: run, --check, --list-devices
│   ├── config.py
│   ├── audio/
│   │   ├── buffer.py          # timestamped ring buffer
│   │   ├── microphone.py
│   │   ├── system_audio.py    # helper subprocess + BlackHole fallback
│   │   └── resample.py
│   ├── asr/
│   │   ├── engine.py          # ASREngine protocol
│   │   ├── whisper_engine.py
│   │   ├── parakeet_engine.py
│   │   ├── vad.py             # Silero ONNX
│   │   ├── segmenter.py
│   │   ├── agreement.py       # LocalAgreement-2
│   │   ├── filters.py         # hallucination guards, cross-source echo dedup
│   │   └── transcription.py   # pipeline orchestration, overload policy
│   ├── metrics.py
│   ├── assistant/
│   │   ├── detector.py        # question / turn-end detection
│   │   ├── providers.py       # AnswerProvider, OpenCode, OpenAI-compatible
│   │   └── practice.py
│   └── web/
│       ├── server.py
│       └── static/            # index.html, app.js, style.css
├── bench/
│   ├── benchmark.py
│   └── data/
└── tests/
    ├── test_buffer.py
    ├── test_agreement.py      # dedup / commit logic
    ├── test_segmenter.py
    ├── test_filters.py
    ├── test_detector.py
    └── test_overload.py
```

## 8. Deliverables

- Working local prototype with mic, system, and both modes.
- `scripts/setup.sh`: checks arm64 and macOS version, creates a venv, installs
  dependencies, builds the Swift helper (if toolchain present), downloads default model
  weights.
- README: permissions (which app gets them, where to toggle), BlackHole + Multi-Output
  Device setup, headphone recommendation, overload policy, measured benchmarks.
- Benchmark script with latency, RTF, memory, and WER.
- Tests for buffer handling, LocalAgreement dedup, segmentation, filters, question
  detection, and overload behavior.
- Optional practice assistant with clear configuration.

## 9. Implementation phases

Each phase ends with its tests passing and a short report: what works, what was measured,
and what could not be tested in this environment.

**Phase 1 — Foundations and risk spike**
- Environment check (`--check`), config, timestamped ring buffer, mic capture.
- MLX Whisper engine with `base.en`, transcribing fixed chunks from the mic.
- **Spike**: prove the system-audio path (build the Swift helper's Process Tap capture,
  or confirm the BlackHole fallback) before going further.
- Tests: buffer.

**Phase 2 — Segmentation and quality**
- Silero ONNX VAD, segmenter with max-window cut, LocalAgreement-2 partials/finals,
  hallucination filters, metrics, overload policy.
- Benchmark script with WER; benchmark all candidate models and pick the default.
- Tests: agreement, segmenter, filters, overload.

**Phase 3 — System audio and dual source**
- Full Swift helper (Process Tap → SCK fallback), BlackHole fallback, `both` mode,
  per-source labels, cross-source echo dedup, reconnect handling.

**Phase 4 — Dashboard**
- FastAPI + WebSocket UI, controls, live metrics, optional transcript persistence.

**Phase 5 — Practice assistant**
- Question detection, practice flow with reveal and feedback, OpenCode and
  OpenAI-compatible providers (with the OpenCode API verified against the installed version).
- Tests: detector, duplicate-request guard.

## Assumptions

- Apple Silicon only. Intel Macs are not supported, since MLX requires Apple Silicon.
- Model weights are downloaded once from Hugging Face on first run, then used offline.
- The Swift helper requires Xcode Command Line Tools. Without them, system audio uses
  BlackHole.
- Benchmark numbers are specific to the machine they are measured on and are reported with
  the chip model.

## Changes made during implementation

- **Phase 1**
  - `sysaudio` re-execs itself with TCC responsibility disclaimed, so the system-audio
    permission belongs to the helper rather than whichever app launched it. Launchers like
    the Claude desktop app hand responsibility to an inner bundle, which was silently
    denied.
- **Phase 2**
  - Added the Apple SpeechAnalyzer backend (`bin/appleasr`, macOS 26). It is genuinely
    streaming and runs out of process.
  - Default model changed from `base.en` to `small.en`, based on the measured benchmark
    (README "Benchmarks").
  - Added a "partial lag" metric for the Whisper and Parakeet pipelines. The Apple backend
    cannot report it, because its volatile results have no per-word timing.
- **Phase 3**
  - The process tap delivers *no frames* while no app plays audio. Stall detection uses a
    helper heartbeat instead, and idle gaps are zero-filled to keep the timeline aligned.
  - Starting a tap blocks while macOS shows the permission dialog. The supervisor waits
    up to 60 s instead of restarting.
- **Phase 5**
  - OpenCode is driven through its documented session API, with deny-all permissions, the
    `plan` agent and an empty working directory.
  - Answers are JSON in the text reply. OpenCode's `json_schema` output is itself a tool,
    which deny-all blocks.
