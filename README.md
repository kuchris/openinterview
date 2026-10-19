# OpenInterview (macOS, local)

Local, low-latency speech recognition for technical-interview practice and meetings where
recording and AI assistance are permitted. Everything runs on-device (MLX on the Apple
Silicon GPU); no audio leaves the machine, and audio is never written to disk.

See [PLAN.md](PLAN.md) for the full design and phase plan.

**Status: Phase 1** — mic capture, fixed-chunk MLX Whisper transcription, latency/memory
metrics, and Core Audio system-audio capture (verified end to end).

## Requirements

- Apple Silicon Mac, macOS 14.2+ (system-audio taps need 14.2; mic works on older versions)
- Xcode or Command Line Tools (for the Swift system-audio helper)
- `ffmpeg` (only for `transcribe-file` / benchmarks): `brew install ffmpeg`

## Install

```bash
scripts/setup.sh
```

This installs dependencies with `uv`, builds `bin/sysaudio`, downloads `base.en`, and runs
`openinterview check`.

## Usage

```bash
uv run openinterview check --probe                       # environment + permission report
uv run openinterview devices                             # list input devices
uv run openinterview run --source mic                    # live mic transcript (Ctrl-C to stop)
uv run openinterview run --source mic --device "MacBook Air"
uv run openinterview run --source system                 # other side of a call
uv run openinterview --model small.en run --source both  # use headphones in this mode
uv run openinterview transcribe-file talk.wav --speed 1  # replay a file through the live pipeline
uv run openinterview --save-transcript run               # opt-in transcript JSONL in transcripts/
```

Models: `tiny.en`, `base.en` (default), `small.en`, `large-v3-turbo`
(all `mlx-community/*` MLX weights, downloaded once to the Hugging Face cache).

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
- `sysaudio` is ad-hoc signed. Rebuilding it may make macOS ask again; if it stops working
  after a rebuild, remove the old `sysaudio` entry in the settings list and re-allow.

**Diagnosing an empty transcript.** If capture runs but no text appears:

- `check --probe` reports all-zero mic input → microphone access is denied.
- The `run --source system` summary shows `"nonzero_frames": 0` while audio is playing →
  system-audio access is denied. A denied tap delivers silence, not an error, so macOS
  gives no other signal.

After changing the microphone permission, quit and reopen the terminal app.

## How it works (Phase 1)

```
mic (PortAudio callback) ─┐                       ┌─ chunker (per source) ─┐
                          ├─► ring buffer (30 s) ─┤                        ├─► bounded queue ─► inference thread (MLX)
sysaudio helper (stdout) ─┘    per source         └────────────────────────┘    drop-oldest       └─► transcript events
```

- **Capture callbacks never touch the model.** They only copy into a fixed-size,
  timestamped ring buffer.
- **System audio** comes from `bin/sysaudio` (Swift, Core Audio Process Tap). It downmixes
  and resamples to 16 kHz natively, and streams framed float32 PCM to Python. Mic and system
  audio stay in separate buffers ("Me" / "Them").
- **Chunked, not streaming.** mlx-whisper transcribes whole buffers. Phase 1 sends fixed 4 s
  chunks and skips silent ones (RMS gate). Phase 2 replaces this with VAD segmentation and
  LocalAgreement partials.
- **Overload policy.** The inference queue holds at most `pipeline.queue_size` jobs. When
  it is full, the **oldest** job is dropped and counted (`dropped_jobs`), so the transcript
  stays near live instead of building a backlog. Capture never blocks.

## Measured results (Phase 1)

Apple M2, 16 GB, macOS 26.6.2, `base.en`, 4 s chunks, file replayed at real-time speed
through the live pipeline:

| Metric | Value |
|---|---|
| ASR inference per 4 s chunk | p50 149 ms, p95 155 ms |
| End-to-end (chunk captured → text) | p50 150 ms, p95 156 ms |
| Real-time factor | 0.037 |
| Mic callback capture latency | ~43 ms (built-in mic, 32 ms blocks) |
| Peak RSS / MLX peak | ~510 MB / ~520 MB |
| Model load (cached) + warmup | ~3 s |

End-to-end here is measured from the end of each chunk. The user-perceived delay also
includes waiting for the chunk to fill (up to 4 s). Phase 2's VAD segmentation and partials
address that. A full multi-model benchmark with WER comes in Phase 2.

Known Phase 1 limitations:

- Fixed chunks can split words at boundaries (e.g. "parent process" → "parent cross").
- No partials or deduplication yet.

Synthetic test audio: macOS's default `say` voice is badly out-of-distribution for small
Whisper models (tiny.en output was nonsense). Use `say -v Samantha` for test clips; it
transcribes verbatim with `base.en`.

## Tests

```bash
uv run pytest
```

Covers the ring buffer (wraparound, overrun accounting, bounded memory, concurrent
reader/writer), the drop-oldest queue, the pipeline with a fake engine (chunking, remainder
flush, silence skip, overload drop accounting, same-thread engine use), and the helper's
frame protocol.
