"""Saved interview sessions: one folder per recording run.

    ~/.openinterview/sessions/20261019-160512-google-backend/
        session.json      title, times, source, model, target, Q&A cards, stats
        transcript.jsonl  finals / utterances / retractions, in order
        mic.m4a           audio per source (optional; AAC via macOS afconvert,
        system.m4a        WAV if afconvert is unavailable)

Everything stays on this machine. Saving (and audio separately) can be turned off in the
config or the app, and any session can be deleted.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def default_dir() -> Path:
    return Path.home() / ".openinterview" / "sessions"


def resolve_dir(path: str | None) -> Path:
    return Path(path).expanduser() if path else default_dir()


def slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:limit].strip("-")


class AudioWriter:
    """Streams one source's ring buffer to a 16-bit WAV on its own thread (crash-safe:
    the WAV is valid up to the last flush), then compresses it when the session ends."""

    def __init__(self, source, path: Path) -> None:
        self.source = source
        self.path = path
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"audio-{source.name}",
                                        daemon=True)
        self.samples = 0

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        buf = self.source.buffer
        pos = buf.total_written
        with wave.open(str(self.path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(buf.sample_rate)
            while True:
                buf.wait_for(pos + buf.sample_rate // 2, timeout=0.5)
                res = buf.read(pos)
                if res.samples.size:
                    pcm = np.clip(res.samples, -1.0, 1.0)
                    w.writeframes((pcm * 32767).astype("<i2").tobytes())
                    self.samples += res.samples.size
                    pos = res.start_index + res.samples.size
                if self._stop.is_set() or (buf.closed and buf.total_written <= pos):
                    break

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)


def compress(wav: Path) -> Path:
    """WAV -> AAC .m4a with macOS afconvert (32 kbps mono, ~15 MB/hour). Keeps the WAV if
    conversion is unavailable or fails."""
    if not wav.exists() or not shutil.which("afconvert"):
        return wav
    m4a = wav.with_suffix(".m4a")
    r = subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "32000", str(wav), str(m4a)],
                       capture_output=True, timeout=600)
    if r.returncode == 0 and m4a.exists() and m4a.stat().st_size > 0:
        wav.unlink()
        return m4a
    log.warning("afconvert failed for %s: %s", wav, r.stderr[-300:])
    return wav


class SessionRecord:
    """The on-disk record of one running session. Thread-safe."""

    def __init__(self, folder: Path, meta: dict) -> None:
        self.folder = folder
        self.meta = meta
        self.cards: dict[int, dict] = {}
        self._lock = threading.Lock()
        self._transcript = (folder / "transcript.jsonl").open("a")
        self._writers: list[AudioWriter] = []
        self._last_save = 0.0
        self.save()

    def start_audio(self, sources: list) -> None:
        for src in sources:
            writer = AudioWriter(src, self.folder / f"{src.name}.wav")
            writer.start()
            self._writers.append(writer)

    def on_message(self, msg: dict) -> None:
        kind = msg.get("type")
        if kind == "transcript":
            ev = msg["event"]
            if ev["kind"] in ("final", "utterance", "retract") and (ev["text"] or ev["kind"] == "retract"):
                with self._lock:
                    self._transcript.write(json.dumps(ev) + "\n")
                    self._transcript.flush()
        elif kind == "assistant_card":
            card = msg["card"]
            with self._lock:
                self.cards[card["id"]] = card
            self.save(throttle=True)

    def save(self, throttle: bool = False) -> None:
        now = time.monotonic()
        if throttle and now - self._last_save < 1.0:
            return
        with self._lock:
            self._last_save = now
            data = {**self.meta, "cards": sorted(self.cards.values(), key=lambda c: c["id"])}
        tmp = self.folder / "session.json.tmp"
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.folder / "session.json")

    def finish(self, stats: dict | None = None) -> None:
        for w in self._writers:
            w.stop()
        audio = {}
        for w in self._writers:
            if w.samples:
                audio[w.source.name] = compress(w.path).name
            elif w.path.exists():
                w.path.unlink()
        with self._lock:
            self._transcript.close()
            self.meta["ended"] = datetime.now().isoformat(timespec="seconds")
            started = datetime.fromisoformat(self.meta["created"])
            self.meta["duration_s"] = round((datetime.now() - started).total_seconds())
            self.meta["audio"] = audio
            if stats:
                self.meta["stats"] = stats
        self.save()


class SessionStore:
    def __init__(self, folder: Path) -> None:
        self.folder = folder

    def create(self, title: str, source: str, model: str, target: dict) -> SessionRecord:
        now = datetime.now()
        name = now.strftime("%Y%m%d-%H%M%S")
        if slug(title):
            name += "-" + slug(title)
        folder = self.folder / name
        folder.mkdir(parents=True, exist_ok=True)
        meta = {"id": name, "title": title or f"Session {now:%b %d, %H:%M}",
                "created": now.isoformat(timespec="seconds"), "ended": None,
                "source": source, "model": model, "target": target, "audio": {}}
        return SessionRecord(folder, meta)

    def _path(self, session_id: str) -> Path:
        if not re.fullmatch(r"[0-9]{8}-[0-9]{6}(-[a-z0-9-]+)?", session_id):
            raise KeyError(session_id)
        path = self.folder / session_id
        if not (path / "session.json").exists():
            raise KeyError(session_id)
        return path

    def list(self) -> list[dict]:
        out = []
        if not self.folder.exists():
            return out
        for d in sorted(self.folder.iterdir(), reverse=True):
            meta_file = d / "session.json"
            if not meta_file.is_file():
                continue
            try:
                meta = json.loads(meta_file.read_text())
            except ValueError:
                continue
            size = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            out.append({k: meta.get(k) for k in ("id", "title", "created", "ended",
                                                  "duration_s", "source", "model", "target",
                                                  "audio")}
                       | {"questions": len(meta.get("cards", [])), "size_bytes": size})
        return out

    def load(self, session_id: str) -> dict:
        path = self._path(session_id)
        meta = json.loads((path / "session.json").read_text())
        lines = []
        tfile = path / "transcript.jsonl"
        if tfile.exists():
            for raw in tfile.read_text().splitlines():
                try:
                    lines.append(json.loads(raw))
                except ValueError:
                    pass
        retracted = {(x["source"], x["segment_id"]) for x in lines if x["kind"] == "retract"}
        meta["transcript"] = [x for x in lines if x["kind"] == "final"
                              and (x["source"], x["segment_id"]) not in retracted]
        meta["audio_paths"] = {k: str(path / v) for k, v in (meta.get("audio") or {}).items()}
        meta["folder"] = str(path)
        return meta

    def delete(self, session_id: str) -> None:
        shutil.rmtree(self._path(session_id))
