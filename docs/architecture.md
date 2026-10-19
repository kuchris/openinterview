# Architecture

[← README](../README.md) · How the pipeline works: capture, VAD, ASR, filters, system audio backends, echo handling.

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
