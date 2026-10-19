"""Latency / accuracy / memory benchmark through the real live pipeline.

    uv run python bench/benchmark.py prepare                  # fetch + synthesize test audio
    uv run python bench/benchmark.py run --models base.en small.en parakeet-v2
    uv run python bench/benchmark.py run --models base.en --no-prompt --speed 1

Dataset (bench/data, audio git-ignored):
  * librispeech: the first N utterances of LibriSpeech dev-clean (hf-internal-testing dummy
    split), real read speech
  * technical:   interview-style sentences full of technical terms, synthesized with
    macOS `say -v Samantha`. Synthetic voice: useful for vocabulary, not for accents.

All clips are joined into one stream with 1 s gaps and replayed at real-time speed (by
default) through FileSource -> VAD -> segmenter -> scheduler -> engine. The latencies are
therefore what the live app would see. Each model runs in a fresh subprocess, so peak
memory is per model.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SR = 16000
GAP_S = 1.0

LIBRISPEECH_ROWS = ("https://datasets-server.huggingface.co/rows?dataset="
                    "hf-internal-testing/librispeech_asr_dummy&config=clean&split=validation"
                    "&offset=0&length={n}")

TECHNICAL = [
    "Can you explain the difference between a process and a thread?",
    "How would you design a rate limiter for a REST API that handles thousands of requests per second?",
    "In Python, the global interpreter lock prevents CPU bound threads from running in parallel.",
    "I would use a hash map to get constant time lookups, which makes the whole algorithm linear.",
    "What happens when you type a URL into the browser and press enter?",
    "When would a B-tree index not help your query in PostgreSQL?",
    "We deployed the service on Kubernetes with a horizontal pod autoscaler and a Redis cache.",
    "Tell me about a time you had to debug a memory leak in production.",
    "An idempotent endpoint returns the same result no matter how many times the request is retried.",
    "How does asyncio differ from multithreading, and when would you choose one over the other?",
    "Could you walk me through how you would reverse a linked list in place?",
    "The trade-off is between consistency and availability, as described by the CAP theorem.",
]


# -- text normalisation and WER -----------------------------------------------------------

_KEEP = re.compile(r"[^a-z0-9' ]+")
# Spelling-only differences that should not count as recognition errors.
_EQUIV = {"mr": "mister", "mrs": "missus", "dr": "doctor", "recognising": "recognizing"}


def normalize(text: str) -> list[str]:
    """Lowercase, drop punctuation, join hyphenated words ("look-ups" == "lookups")."""
    text = re.sub(r"(?<=\w)-(?=\w)", "", text.lower())
    text = _KEEP.sub(" ", text)
    words = [w.strip("'") for w in text.split() if w.strip("'")]
    return [_EQUIV.get(w, w) for w in words]


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    ref_words = [w for r in refs for w in normalize(r)]
    hyp_words = [w for h in hyps for w in normalize(h)]
    return edit_distance(ref_words, hyp_words) / max(1, len(ref_words))


# -- prepare ------------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> int:
    from openinterview.audio.file_source import load_audio

    DATA.mkdir(parents=True, exist_ok=True)
    manifest = []
    with urllib.request.urlopen(LIBRISPEECH_ROWS.format(n=args.librispeech), timeout=60) as r:
        rows = json.load(r)["rows"]
    for row in rows:
        item = row["row"]
        path = DATA / f"ls_{item['id']}.flac"
        if not path.exists():
            with urllib.request.urlopen(item["audio"][0]["src"], timeout=60) as r:
                path.write_bytes(r.read())
        manifest.append({"subset": "librispeech", "path": path.name, "text": item["text"]})
    for i, sentence in enumerate(TECHNICAL):
        path = DATA / f"tech_{i:02d}.aiff"
        if not path.exists():
            subprocess.run(["say", "-v", args.voice, "-o", str(path), sentence], check=True)
        manifest.append({"subset": "technical", "path": path.name, "text": sentence})
    for m in manifest:
        m["seconds"] = round(load_audio(str(DATA / m["path"])).shape[0] / SR, 2)
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(m["seconds"] for m in manifest)
    print(f"prepared {len(manifest)} clips, {total:.0f} s of speech -> {DATA / 'manifest.json'}")
    return 0


def build_stream(manifest: list[dict]) -> tuple[np.ndarray, list[tuple[float, float]]]:
    from openinterview.audio.file_source import load_audio

    gap = np.zeros(int(GAP_S * SR), np.float32)
    parts, spans, t = [gap], [], GAP_S
    for m in manifest:
        a = load_audio(str(DATA / m["path"]))
        spans.append((t, t + a.shape[0] / SR))
        parts += [a, gap]
        t += a.shape[0] / SR + GAP_S
    return np.concatenate(parts), spans


# -- one model (runs in a subprocess) -----------------------------------------------------

def cmd_one(args: argparse.Namespace) -> int:
    from openinterview.asr.factory import build_transcriber
    from openinterview.audio.buffer import AudioRingBuffer
    from openinterview.audio.file_source import FileSource
    from openinterview.config import load_config
    from openinterview.metrics import peak_memory_mb

    manifest = json.loads((DATA / "manifest.json").read_text())
    audio, spans = build_stream(manifest)
    cfg = load_config()
    cfg.asr.model = args.model
    if args.no_prompt:
        cfg.asr.vocabulary_prompt = ""
        cfg.asr.prompt_with_context = False

    finals: list = []
    t_stream_start = [0.0]

    def on_event(ev):
        if ev.kind == "final" and ev.text:
            finals.append(ev)

    src = FileSource(AudioRingBuffer(60, SR), audio, speed=args.speed, tail_silence_s=1.5)
    pipe = build_transcriber(cfg, [src], on_event)
    t0 = time.monotonic()
    pipe.start()
    pipe.ready.wait()
    if pipe.load_error:
        print(json.dumps({"model": args.model, "error": str(pipe.load_error)}))
        return 1
    load_s = getattr(pipe, "load_seconds", time.monotonic() - t0)
    t_stream_start[0] = time.monotonic() - pipe.session_start
    src.start()
    src.done.wait()
    pipe.stop(drain=True)

    # Assign each final to the clip whose span contains its midpoint (stream time).
    hyps: list[list[str]] = [[] for _ in manifest]
    offset = t_stream_start[0]
    for ev in finals:
        mid = (ev.start + ev.end) / 2 - offset
        if args.speed > 0:
            mid *= args.speed
        idx = min(range(len(spans)), key=lambda i: 0 if spans[i][0] <= mid <= spans[i][1]
                  else min(abs(mid - spans[i][0]), abs(mid - spans[i][1])))
        hyps[idx].append(ev.text)

    out = {"model": args.model, "prompt": not args.no_prompt, "speed": args.speed,
           "model_load_s": round(load_s, 2), "audio_s": round(audio.shape[0] / SR, 1)}
    for subset in ("librispeech", "technical"):
        ids = [i for i, m in enumerate(manifest) if m["subset"] == subset]
        out[f"wer_{subset}"] = round(wer([manifest[i]["text"] for i in ids],
                                         [" ".join(hyps[i]) for i in ids]), 4)
    out["wer_all"] = round(wer([m["text"] for m in manifest], [" ".join(h) for h in hyps]), 4)
    out["pipeline"] = pipe.stats()
    out["memory"] = peak_memory_mb()
    if args.dump:
        out["transcripts"] = [{"ref": m["text"], "hyp": " ".join(h)}
                              for m, h in zip(manifest, hyps)]
    print(json.dumps(out))
    return 0


# -- orchestrate ----------------------------------------------------------------------------

def _chip() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return platform.machine()


def cmd_run(args: argparse.Namespace) -> int:
    if not (DATA / "manifest.json").exists():
        print("run `bench/benchmark.py prepare` first", file=sys.stderr)
        return 1
    results = []
    for model in args.models:
        cmd = [sys.executable, __file__, "one", "--model", model, "--speed", str(args.speed)]
        if args.no_prompt:
            cmd.append("--no-prompt")
        if args.dump:
            cmd.append("--dump")
        print(f"== {model} ...", file=sys.stderr, flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("{")), None)
        if line is None:
            print(proc.stderr[-2000:], file=sys.stderr)
            continue
        r = json.loads(line)
        results.append(r)
        p = r.get("pipeline", {})
        print(f"   WER libri {r.get('wer_librispeech')}, tech {r.get('wer_technical')}; "
              f"final p50 {p.get('final_latency_ms', {}).get('p50')} ms; "
              f"mem {r.get('memory')}", file=sys.stderr, flush=True)

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {"date": stamp, "chip": _chip(), "macos": platform.mac_ver()[0],
            "python": platform.python_version(), "speed": args.speed}
    (RESULTS / f"{stamp}.json").write_text(json.dumps({"meta": meta, "results": results},
                                                      indent=2))
    md = render_markdown(meta, results)
    (RESULTS / f"{stamp}.md").write_text(md)
    print(md)
    return 0


def render_markdown(meta: dict, results: list[dict]) -> str:
    def g(r, *keys, default="-"):
        cur = r
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    lines = [
        f"# Benchmark {meta['date']}",
        "",
        f"{meta['chip']}, macOS {meta['macos']}, Python {meta['python']}, "
        f"replay speed x{meta['speed']}",
        "",
        "| Model | Prompt | WER libri | WER tech | First partial p50 | Partial lag p50 "
        "| Final after speech end p50 / p95 | ASR final p50 | ASR partial p50 | RTF p50 "
        "| VAD µs/frame p50 | RSS peak | MLX peak | Load |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['model']} | error: {r['error']} |" + " |" * 12)
            continue
        lines.append(
            f"| {r['model']} | {'on' if r['prompt'] else 'off'} "
            f"| {r['wer_librispeech'] * 100:.1f}% | {r['wer_technical'] * 100:.1f}% "
            f"| {g(r, 'pipeline', 'first_partial_ms', 'p50')} ms "
            f"| {g(r, 'pipeline', 'partial_lag_ms', 'p50')} ms "
            f"| {g(r, 'pipeline', 'final_latency_ms', 'p50')} / "
            f"{g(r, 'pipeline', 'final_latency_ms', 'p95')} ms "
            f"| {g(r, 'pipeline', 'inference_final_ms', 'p50')} ms "
            f"| {g(r, 'pipeline', 'inference_partial_ms', 'p50')} ms "
            f"| {g(r, 'pipeline', 'rtf', 'p50')} "
            f"| {g(r, 'pipeline', 'vad_frame_us', 'p50')} "
            f"| {g(r, 'memory', 'rss_peak_mb')} MB"
            + (f" (+helper {g(r, 'pipeline', 'helper_rss_mb')} MB)"
               if g(r, 'pipeline', 'helper_rss_mb') != "-" else "")
            + f" | {g(r, 'memory', 'mlx_peak_mb')} MB "
            f"| {r['model_load_s']} s |"
        )
    lines += [
        "",
        "Overload counters (finals dropped / partials superseded / partials skipped):",
        "",
    ]
    for r in results:
        if "error" not in r:
            p = r["pipeline"]
            lines.append(f"- {r['model']}: {p['finals_dropped']} / {p['partials_superseded']} / "
                         f"{p['partials_skipped']}; segments {p['segments']}, "
                         f"forced cuts {p['forced_cuts']}, filtered {p['filtered']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--librispeech", type=int, default=12)
    p.add_argument("--voice", default="Samantha")
    p.set_defaults(func=cmd_prepare)
    for name, func in (("run", cmd_run), ("one", cmd_one)):
        q = sub.add_parser(name)
        if name == "run":
            q.add_argument("--models", nargs="+",
                           default=["tiny.en", "base.en", "small.en", "large-v3-turbo",
                                    "parakeet-v2"])
        else:
            q.add_argument("--model", required=True)
        q.add_argument("--speed", type=float, default=1.0)
        q.add_argument("--no-prompt", action="store_true")
        q.add_argument("--dump", action="store_true", help="include per-clip transcripts")
        q.set_defaults(func=func)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
