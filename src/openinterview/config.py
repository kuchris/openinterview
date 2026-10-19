"""Configuration loading: config.yaml merged with CLI overrides."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class AudioConfig:
    source: str = "mic"
    mic_device: int | str | None = None
    sample_rate: int = 16000
    ring_buffer_seconds: float = 30.0


@dataclass
class ASRConfig:
    model: str = "base.en"
    language: str = "en"


@dataclass
class PipelineConfig:
    chunk_seconds: float = 4.0
    silence_rms: float = 0.004
    queue_size: int = 4


@dataclass
class StorageConfig:
    save_transcripts: bool = False
    transcript_dir: str = "transcripts"


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)


def _apply(obj: Any, data: dict[str, Any]) -> None:
    known = {f.name: f for f in fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ValueError(f"Unknown config key: {key!r}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _apply(current, value)
        else:
            setattr(obj, key, value)


def load_config(path: Path | None = None) -> Config:
    cfg = Config()
    path = path or DEFAULT_CONFIG_PATH
    if path.exists():
        data = yaml.safe_load(path.read_text()) or {}
        _apply(cfg, data)
    return cfg
