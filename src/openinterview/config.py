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
    system_backend: str = "auto"   # auto | tap | blackhole
    blackhole_device: str = "BlackHole"
    sample_rate: int = 16000
    ring_buffer_seconds: float = 30.0


@dataclass
class ASRConfig:
    model: str = "small.en"
    language: str = "en"
    vocabulary_prompt: str = ""
    prompt_with_context: bool = True


@dataclass
class PipelineConfig:
    vad: str = "silero"            # silero | energy
    vad_threshold: float = 0.5
    min_silence_s: float = 0.6
    max_segment_s: float = 15.0
    partial_interval_s: float = 1.0
    max_finals: int = 8
    degrade_at: int = 2
    merge_gap_s: float = 1.2
    echo_suppression: bool = True


@dataclass
class StorageConfig:
    save_sessions: bool = True         # each run saved under sessions_dir (text)
    save_audio: bool = True            # ... plus per-source audio (.m4a)
    sessions_dir: str | None = None    # default ~/.openinterview/sessions


@dataclass
class AssistantConfig:
    enabled: bool = False
    provider: str = "opencode"           # opencode | openai
    base_url: str = "http://127.0.0.1:4096"
    model: str | None = None             # opencode: "providerID/modelID"; openai: model name
    api_key_env: str = "OPENAI_API_KEY"  # openai provider only; key read from this env var
    mode: str = "immediate"              # immediate | practice (hide until revealed)
    listen_to: list[str] = field(default_factory=lambda: ["auto"])  # auto | [system, mic]
    filter_small_talk: bool = True    # only interview questions (technical/behavioural)
    profile_path: str | None = None   # personal profile/CV (default ~/.openinterview/profile.md)
    timeout_s: float = 60.0
    max_cards: int = 50


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    assistant: AssistantConfig = field(default_factory=AssistantConfig)


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
