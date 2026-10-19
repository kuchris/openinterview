"""CLI entry point.

  openinterview check [--probe]          environment / permission report
  openinterview devices                  list input devices
  openinterview run [--source mic|system|both] [--model base.en] [--duration S]
  openinterview transcribe-file FILE [--speed 1.0] [--model base.en]
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import signal
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from .config import PROJECT_ROOT, Config, load_config

SPEAKER = {"mic": "Me", "system": "Them"}


# -- check ---------------------------------------------------------------------------------

def _model_cached(repo: str) -> bool:
    try:
        from huggingface_hub import constants
        cache = Path(constants.HF_HUB_CACHE)
    except Exception:
        cache = Path.home() / ".cache" / "huggingface" / "hub"
    return (cache / f"models--{repo.replace('/', '--')}").exists()


def _probe_mic(seconds: float = 0.6) -> str:
    import numpy as np
    import sounddevice as sd

    rec = sd.rec(int(16000 * seconds), samplerate=16000, channels=1, dtype="float32")
    sd.wait()
    peak = float(np.abs(rec).max())
    if peak == 0.0:
        return "all-zero input: permission likely denied, or input muted"
    return f"receiving audio (peak {peak:.3f})"


def cmd_check(args: argparse.Namespace, cfg: Config) -> int:
    from .asr.whisper_engine import MODEL_REPOS
    from .audio.system_audio import HELPER_PATH, helper_available, helper_permissions

    mac_ver = platform.mac_ver()[0]
    report: dict = {
        "platform": sys.platform,
        "machine": platform.machine(),
        "macos": mac_ver,
        "python": platform.python_version(),
        "ffmpeg": shutil.which("ffmpeg") or "missing (needed only for transcribe-file)",
    }
    problems: list[str] = []
    if sys.platform != "darwin" or platform.machine() != "arm64":
        problems.append("Apple Silicon macOS is required (MLX runs on Metal)")
    try:
        major, minor = (int(x) for x in mac_ver.split(".")[:2])
        report["process_tap_supported"] = (major, minor) >= (14, 2)
    except ValueError:
        report["process_tap_supported"] = False

    try:
        import mlx.core as mx

        report["mlx"] = mx.__version__
        report["mlx_device"] = str(mx.default_device())
        report["metal_available"] = bool(mx.metal.is_available())
    except Exception as exc:
        problems.append(f"mlx import failed: {exc}")

    report["models_cached"] = {name: _model_cached(repo) for name, repo in MODEL_REPOS.items()}

    try:
        from .audio.microphone import list_input_devices

        devices = list_input_devices()
        report["default_input"] = next((d["name"] for d in devices if d["is_default"]), None)
        report["blackhole_installed"] = any("blackhole" in d["name"].lower() for d in devices)
    except Exception as exc:
        problems.append(f"PortAudio failed: {exc}")

    report["sysaudio_helper"] = str(HELPER_PATH) if helper_available() else "not built"
    if helper_available():
        try:
            report["permissions"] = helper_permissions()
        except Exception as exc:
            report["permissions"] = f"query failed: {exc}"
    else:
        problems.append("system audio helper missing: run scripts/build_helper.sh")

    if args.probe:
        try:
            report["mic_probe"] = _probe_mic()
        except Exception as exc:
            report["mic_probe"] = f"failed: {exc}"

    print(json.dumps(report, indent=2))
    for p in problems:
        print(f"PROBLEM: {p}", file=sys.stderr)
    print(
        "\nPermissions are granted to the app that launches Python (Terminal, iTerm2, VS Code...):\n"
        "  Microphone:   System Settings > Privacy & Security > Microphone\n"
        "  System audio: System Settings > Privacy & Security > Screen & System Audio Recording\n"
        "                (\"System Audio Recording Only\" list)",
        file=sys.stderr,
    )
    return 1 if problems else 0


def cmd_devices(args: argparse.Namespace, cfg: Config) -> int:
    from .audio.microphone import list_input_devices

    for d in list_input_devices():
        mark = "*" if d["is_default"] else " "
        print(f"{mark} [{d['index']}] {d['name']}  ({d['channels']} ch, {d['default_samplerate']:.0f} Hz)")
    return 0


# -- run -------------------------------------------------------------------------------

class TranscriptPrinter:
    def __init__(self, save_dir: Path | None) -> None:
        self._lock = threading.Lock()
        self._file = None
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
            self._file = path.open("a")
            print(f"saving transcript to {path}", file=sys.stderr)

    def __call__(self, ev) -> None:
        with self._lock:
            print(f"[{ev.start:7.1f}-{ev.end:7.1f}] {SPEAKER.get(ev.source, ev.source):>4}: {ev.text}"
                  f"   (e2e {ev.latency_s * 1000:.0f} ms, asr {ev.inference_s * 1000:.0f} ms)",
                  flush=True)
            if self._file:
                self._file.write(json.dumps(ev.__dict__) + "\n")
                self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


def _status(name, state, detail) -> None:
    print(f"  · {name}: {state.value}{' — ' + detail if detail else ''}", file=sys.stderr)


def _run_pipeline(cfg: Config, sources: list, duration: float | None,
                  until=None) -> int:
    from .asr.transcription import ChunkedTranscriber
    from .asr.whisper_engine import WhisperEngine
    from .metrics import peak_memory_mb

    save_dir = PROJECT_ROOT / cfg.storage.transcript_dir if cfg.storage.save_transcripts else None
    printer = TranscriptPrinter(save_dir)
    pipe = ChunkedTranscriber(
        engine_factory=lambda: WhisperEngine(cfg.asr.model, cfg.asr.language),
        sources=sources,
        on_event=printer,
        chunk_seconds=cfg.pipeline.chunk_seconds,
        silence_rms=cfg.pipeline.silence_rms,
        queue_size=cfg.pipeline.queue_size,
    )
    print(f"loading {cfg.asr.model} ...", file=sys.stderr)
    t0 = time.monotonic()
    pipe.start()
    pipe.ready.wait()
    if pipe.load_error:
        print(f"model load failed: {pipe.load_error}", file=sys.stderr)
        return 1
    print(f"model ready in {time.monotonic() - t0:.1f} s; capturing (Ctrl-C to stop)", file=sys.stderr)

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    started = []
    try:
        for src in sources:
            src.start()
            started.append(src)
        deadline = time.monotonic() + duration if duration else None
        while not stop.is_set():
            if deadline and time.monotonic() >= deadline:
                break
            if until is not None and until():
                break
            stop.wait(0.1)
    except Exception as exc:
        print(f"capture failed: {exc}", file=sys.stderr)
    finally:
        for src in started:
            src.stop()
        pipe.stop(drain=True)
        printer.close()

    summary = {
        "model": cfg.asr.model,
        "pipeline": pipe.stats(),
        "sources": {s.name: s.stats() for s in sources},
        "memory": peak_memory_mb(),
    }
    print(json.dumps(summary, indent=2), file=sys.stderr)
    return 0


def _make_sources(cfg: Config, source: str) -> list:
    from .audio.buffer import AudioRingBuffer
    from .audio.microphone import MicrophoneSource
    from .audio.system_audio import SystemAudioSource

    sr, secs = cfg.audio.sample_rate, cfg.audio.ring_buffer_seconds
    sources = []
    if source in ("mic", "both"):
        sources.append(MicrophoneSource(AudioRingBuffer(secs, sr), cfg.audio.mic_device,
                                        on_status=_status))
    if source in ("system", "both"):
        sources.append(SystemAudioSource(AudioRingBuffer(secs, sr), on_status=_status))
    if source == "both":
        print("note: use headphones in 'both' mode, or the mic will re-capture speaker audio",
              file=sys.stderr)
    return sources


def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    return _run_pipeline(cfg, _make_sources(cfg, cfg.audio.source), args.duration)


def cmd_transcribe_file(args: argparse.Namespace, cfg: Config) -> int:
    from .audio.buffer import AudioRingBuffer
    from .audio.file_source import FileSource, load_audio

    audio = load_audio(args.file, cfg.audio.sample_rate)
    print(f"{args.file}: {audio.shape[0] / cfg.audio.sample_rate:.1f} s, speed x{args.speed}",
          file=sys.stderr)
    src = FileSource(AudioRingBuffer(cfg.audio.ring_buffer_seconds, cfg.audio.sample_rate),
                     audio, speed=args.speed)
    return _run_pipeline(cfg, [src], None, until=src.done.is_set)


# -- entry -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openinterview", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model", help="tiny.en | base.en | small.en | large-v3-turbo")
    p.add_argument("--save-transcript", action="store_true", help="write transcript JSONL")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("check", help="report environment and permissions")
    c.add_argument("--probe", action="store_true", help="record 0.6 s from the mic to test access")
    c.set_defaults(func=cmd_check)

    sub.add_parser("devices", help="list input devices").set_defaults(func=cmd_devices)

    r = sub.add_parser("run", help="live transcription")
    r.add_argument("--source", choices=["mic", "system", "both"])
    r.add_argument("--device", help="mic device index or name substring")
    r.add_argument("--duration", type=float, help="stop after N seconds")
    r.set_defaults(func=cmd_run)

    f = sub.add_parser("transcribe-file", help="replay a file through the live pipeline")
    f.add_argument("file")
    f.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = max speed")
    f.set_defaults(func=cmd_transcribe_file)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if args.model:
        cfg.asr.model = args.model
    if args.save_transcript:
        cfg.storage.save_transcripts = True
    if getattr(args, "source", None):
        cfg.audio.source = args.source
    if getattr(args, "device", None):
        cfg.audio.mic_device = args.device
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
