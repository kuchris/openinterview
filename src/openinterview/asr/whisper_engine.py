"""MLX Whisper backend (Apple Silicon GPU via Metal).

mlx-whisper transcribes a complete buffer per call; there is no incremental decoder. The
pipeline therefore feeds it chunks/segments (chunked transcription, not native streaming).

MLX keeps per-thread GPU streams, so `load`, `warmup` and `transcribe` must all run on the
same thread: the pipeline's inference thread.
"""

from __future__ import annotations

import time

import numpy as np

from .engine import ASRResult, Segment

MODEL_REPOS = {
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "small": "mlx-community/whisper-small-mlx",  # multilingual (Japanese, Mandarin)
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


class WhisperEngine:
    def __init__(self, model: str = "base.en", language: str = "en") -> None:
        if model not in MODEL_REPOS and "/" not in model:
            raise ValueError(f"Unknown model {model!r}; choose from {sorted(MODEL_REPOS)}")
        self.model = model
        self.repo = MODEL_REPOS.get(model, model)
        self.language = language
        self.name = f"whisper-{model}"
        self._transcribe = None

    def load(self) -> None:
        import importlib

        import mlx.core as mx

        # The package re-exports a `transcribe` function that shadows the submodule name.
        mod = importlib.import_module("mlx_whisper.transcribe")

        mod.ModelHolder.get_model(self.repo, mx.float16)  # downloads on first use, then cached
        self._transcribe = mod.transcribe

    def warmup(self) -> None:
        # First call compiles Metal kernels; do it before real audio arrives.
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult:
        if self._transcribe is None:
            self.load()
        t0 = time.perf_counter()
        out = self._transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=self.language,
            temperature=0.0,                  # no fallback re-decodes: predictable latency
            condition_on_previous_text=False,  # avoids hallucination loops across chunks
            initial_prompt=prompt,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            verbose=None,
        )
        elapsed = time.perf_counter() - t0
        segments = [
            Segment(
                text=s["text"].strip(),
                start=float(s["start"]),
                end=float(s["end"]),
                avg_logprob=float(s.get("avg_logprob", 0.0)),
                no_speech_prob=float(s.get("no_speech_prob", 0.0)),
                compression_ratio=float(s.get("compression_ratio", 0.0)),
            )
            for s in out.get("segments", [])
            if s["text"].strip()
        ]
        return ASRResult(
            text=" ".join(s.text for s in segments),
            segments=segments,
            audio_seconds=audio.shape[0] / 16000,
            inference_seconds=elapsed,
        )
