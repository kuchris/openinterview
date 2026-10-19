"""CLI entry point.

  openinterview check [--probe]          environment / permission report
  openinterview devices                  list input devices
  openinterview run [--source mic|system|both] [--model base.en] [--duration S]
  openinterview transcribe-file FILE [--speed 1.0] [--model base.en]
  openinterview serve [--port 8765]           local web dashboard
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import shutil
import signal
import subprocess
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
    from .asr.factory import MODELS as MODEL_REPOS
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

    report["models_cached"] = {name: _model_cached(repo) for name, repo in MODEL_REPOS.items()
                               if "/" in repo}
    from .asr.apple_stream import HELPER_PATH as APPLE_HELPER

    if APPLE_HELPER.is_file():
        try:
            out = subprocess.run([str(APPLE_HELPER), "status"], capture_output=True, text=True,
                                 timeout=30).stdout
            report["apple_speech"] = json.loads(out.strip().splitlines()[-1])
        except Exception as exc:
            report["apple_speech"] = f"status failed: {exc}"
    else:
        report["apple_speech"] = "helper not built (macOS 26+)"
    try:
        import onnxruntime

        from .asr.vad import CACHE_DIR

        report["onnxruntime"] = onnxruntime.__version__
        report["silero_vad_cached"] = any(CACHE_DIR.glob("silero_vad_*.onnx"))
    except Exception as exc:
        problems.append(f"onnxruntime unavailable ({exc}); set pipeline.vad: energy")

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
    from .audio.system_audio import BLACKHOLE_SETUP

    tap_ok = helper_available() and report.get("process_tap_supported")
    report["system_audio_backend"] = ("tap" if tap_ok else
                                      "blackhole" if report.get("blackhole_installed") else "none")
    if report["system_audio_backend"] == "none":
        problems.append("no system audio path: build the helper (scripts/build_helper.sh, "
                        "macOS 14.2+) or install BlackHole\n" + BLACKHOLE_SETUP)

    a = cfg.assistant
    try:
        from .assistant.practice import make_provider

        ok, detail = make_provider(a).health()
        report["assistant"] = {"enabled": a.enabled, "provider": a.provider,
                               "reachable": ok, "detail": detail}
    except Exception as exc:
        report["assistant"] = {"enabled": a.enabled, "provider": a.provider,
                               "reachable": False, "detail": str(exc)}

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
    """Console view. Finals and utterances print as lines; on a TTY the current partial is
    redrawn in grey on the line below the last final."""

    GREY, RESET, CLEAR = "\033[90m", "\033[0m", "\r\033[K"

    def __init__(self, save_dir: Path | None, show_partials: bool | None = None,
                 show_utterances: bool = False) -> None:
        self._lock = threading.Lock()
        self._file = None
        self.tty = sys.stdout.isatty()
        self.show_partials = self.tty if show_partials is None else show_partials
        self.show_utterances = show_utterances
        self._partial_shown = False
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"{datetime.now():%Y%m%d-%H%M%S}.jsonl"
            self._file = path.open("a")
            print(f"saving transcript to {path}", file=sys.stderr)

    def _clear_partial(self) -> None:
        if self._partial_shown and self.tty:
            sys.stdout.write(self.CLEAR)
        self._partial_shown = False

    def __call__(self, ev) -> None:
        who = SPEAKER.get(ev.source, ev.source)
        with self._lock:
            if ev.kind == "partial":
                if not self.show_partials:
                    return
                if self.tty:
                    self._clear_partial()
                    sys.stdout.write(f"{self.GREY}{who:>4}: {ev.stable} {ev.unstable}{self.RESET}"[:400])
                    self._partial_shown = True
                else:
                    print(f"  ~ {who:>4}: {ev.stable} | {ev.unstable}")
                sys.stdout.flush()
                return
            self._clear_partial()
            if ev.kind == "final" and ev.text:
                print(f"[{ev.start:7.1f}-{ev.end:7.1f}] {who:>4} #{ev.segment_id}: {ev.text}"
                      f"   ({ev.latency_s * 1000:.0f} ms after speech end,"
                      f" asr {ev.inference_s * 1000:.0f} ms)", flush=True)
            elif ev.kind == "retract":
                print(f"  (removed Me #{ev.segment_id}: it was the speakers' echo of Them)",
                      flush=True)
            elif ev.kind == "utterance" and self.show_utterances:
                print(f"  => utterance {who}: {ev.text}", flush=True)
            if self._file and (ev.kind == "retract" or (ev.text and ev.kind in ("final", "utterance"))):
                self._file.write(json.dumps(ev.to_dict()) + "\n")
                self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._clear_partial()
        if self._file:
            self._file.close()


def _status(name, state, detail) -> None:
    print(f"  · {name}: {state.value}{' — ' + detail if detail else ''}", file=sys.stderr)


def _run_pipeline(cfg: Config, sources: list, duration: float | None,
                  until=None, printer: TranscriptPrinter | None = None) -> int:
    from .asr.factory import build_transcriber
    from .metrics import peak_memory_mb

    save_dir = PROJECT_ROOT / cfg.storage.transcript_dir if cfg.storage.save_transcripts else None
    printer = printer or TranscriptPrinter(save_dir)
    pipe = build_transcriber(cfg, sources, printer)
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
    from .audio.microphone import resolve_device, sd
    from .audio.system_audio import AutoSystemSource

    sr, secs = cfg.audio.sample_rate, cfg.audio.ring_buffer_seconds
    sources = []
    if source in ("mic", "both"):
        idx = resolve_device(cfg.audio.mic_device)
        name = sd.query_devices(idx if idx is not None else sd.default.device[0])["name"]
        if "blackhole" in name.lower():
            raise SystemExit(f"mic device is {name!r}, a loopback device: it would transcribe "
                             "system audio as 'Me'. Pick a real mic with --device.")
        sources.append(MicrophoneSource(AudioRingBuffer(secs, sr), cfg.audio.mic_device,
                                        on_status=_status))
    if source in ("system", "both"):
        sources.append(AutoSystemSource(AudioRingBuffer(secs, sr), cfg.audio.system_backend,
                                        cfg.audio.blackhole_device, on_status=_status))
    if source == "both":
        print("note: use headphones in 'both' mode, or the mic will re-capture speaker audio",
              file=sys.stderr)
    return sources


def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    printer = TranscriptPrinter(
        PROJECT_ROOT / cfg.storage.transcript_dir if cfg.storage.save_transcripts else None,
        show_partials=True if args.partials else None, show_utterances=args.utterances)
    return _run_pipeline(cfg, _make_sources(cfg, cfg.audio.source), args.duration,
                         printer=printer)


def cmd_transcribe_file(args: argparse.Namespace, cfg: Config) -> int:
    from .audio.buffer import AudioRingBuffer
    from .audio.file_source import FileSource, load_audio

    audio = load_audio(args.file, cfg.audio.sample_rate)
    print(f"{args.file}: {audio.shape[0] / cfg.audio.sample_rate:.1f} s, speed x{args.speed}",
          file=sys.stderr)
    src = FileSource(AudioRingBuffer(cfg.audio.ring_buffer_seconds, cfg.audio.sample_rate),
                     audio, speed=args.speed)
    printer = TranscriptPrinter(None, show_partials=True if args.partials else None,
                                show_utterances=args.utterances)
    return _run_pipeline(cfg, [src], None, until=src.done.is_set, printer=printer)


# -- entry -----------------------------------------------------------------------------

def cmd_serve(args: argparse.Namespace, cfg: Config) -> int:
    from .web.server import serve

    serve(cfg, port=args.port)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="openinterview", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--model",
                   help="tiny.en | base.en | small.en | large-v3-turbo | parakeet-v2 | apple")
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
    r.add_argument("--system-backend", choices=["auto", "tap", "blackhole"])
    r.add_argument("--duration", type=float, help="stop after N seconds")
    r.add_argument("--partials", action="store_true", help="print partials even when not a TTY")
    r.add_argument("--utterances", action="store_true", help="print merged utterances")
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("serve", help="local web dashboard (127.0.0.1)")
    w.add_argument("--port", type=int, default=8765)
    w.set_defaults(func=cmd_serve)

    f = sub.add_parser("transcribe-file", help="replay a file through the live pipeline")
    f.add_argument("file")
    f.add_argument("--speed", type=float, default=1.0, help="1 = real time, 0 = max speed")
    f.add_argument("--partials", action="store_true", help="print partials even when not a TTY")
    f.add_argument("--utterances", action="store_true", help="print merged utterances")
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
    if getattr(args, "system_backend", None):
        cfg.audio.system_backend = args.system_backend
    return args.func(args, cfg)


if __name__ == "__main__":
    sys.exit(main())
