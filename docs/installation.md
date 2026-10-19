# Installation

[← README](../README.md) · Requirements, setup and the macOS permissions OpenInterview needs.

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
2. builds the Swift helpers in `bin/`: `sysaudio` (system audio), `appleasr` (Apple
   speech recognition), `pdftext` (CV PDFs) and `openinterview-ui` (the app window);
3. downloads `small.en` and the Silero VAD model (2 MB, SHA-256 pinned);
4. runs `openinterview check`.

For Parakeet, also run `uv sync --extra parakeet`.

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
