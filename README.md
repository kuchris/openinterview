# OpenInterview (macOS, local)

Local, low-latency speech recognition for technical-interview practice and meetings where
recording and AI assistance are permitted. Everything runs on-device (MLX on the Apple
Silicon GPU); no audio leaves the machine, and audio is never written to disk.

See [PLAN.md](PLAN.md) for the full design and phase plan.

**Status: Phase 2.** Working so far:

- Mic and system-audio capture.
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
uv run openinterview check --probe                       # environment + permission report
uv run openinterview devices                             # list input devices
uv run openinterview run --source mic                    # live mic transcript (Ctrl-C to stop)
uv run openinterview run --source mic --device "MacBook Air"
uv run openinterview run --source system                 # other side of a call
uv run openinterview --model base.en run --source both   # use headphones in this mode
uv run openinterview --model apple run --source system   # Apple on-device streaming ASR
uv run openinterview transcribe-file talk.wav --speed 1 --utterances  # replay a file live
uv run openinterview --save-transcript run               # opt-in transcript JSONL in transcripts/
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
- **Helper frame protocol.**
- **Benchmark WER.**
