# Testing

[← README](../README.md) · The mock interview end-to-end test and the unit test suite.

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

## Unit tests

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
