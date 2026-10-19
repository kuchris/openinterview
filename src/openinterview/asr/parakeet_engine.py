"""NVIDIA Parakeet TDT (0.6B) via parakeet-mlx. Optional: `uv sync --extra parakeet`.

English-only (v2) or multilingual (v3) transducer. It transcribes a full buffer per call,
like the Whisper backend. It has no prompt / vocabulary biasing.
Same-thread rule as Whisper: load and transcribe on the inference thread.
"""

from __future__ import annotations

import time

import numpy as np

from .engine import ASRResult, Segment

MODEL_REPOS = {
    "parakeet-v2": "mlx-community/parakeet-tdt-0.6b-v2",
    "parakeet-v3": "mlx-community/parakeet-tdt-0.6b-v3",
}


class ParakeetEngine:
    def __init__(self, model: str = "parakeet-v2") -> None:
        self.model = model
        self.repo = MODEL_REPOS.get(model, model)
        self.name = model
        self._model = None

    def load(self) -> None:
        try:
            from parakeet_mlx import from_pretrained
        except ImportError as exc:
            raise RuntimeError("parakeet-mlx not installed: run `uv sync --extra parakeet`") from exc
        self._model = from_pretrained(self.repo)

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))

    def transcribe(self, audio: np.ndarray, prompt: str | None = None) -> ASRResult:
        import mlx.core as mx
        from parakeet_mlx.audio import get_logmel

        if self._model is None:
            self.load()
        t0 = time.perf_counter()
        # get_logmel must get float32: it reinterprets the complex STFT via mx.view(x, dtype).
        mel = get_logmel(mx.array(audio, dtype=mx.float32), self._model.preprocessor_config)
        result = self._model.generate(mel)[0]
        elapsed = time.perf_counter() - t0
        segments = []
        for s in result.sentences:
            text = s.text.strip()
            if not text:
                continue
            # Map mean token confidence onto Whisper-style fields so the shared filters work.
            conf = max(1e-6, float(s.confidence))
            segments.append(Segment(text, float(s.start), float(s.end),
                                    avg_logprob=float(np.log(conf)), no_speech_prob=0.0,
                                    compression_ratio=0.0))
        return ASRResult(
            text=" ".join(s.text for s in segments),
            segments=segments,
            audio_seconds=audio.shape[0] / 16000,
            inference_seconds=elapsed,
        )
