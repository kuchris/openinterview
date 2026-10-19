"""Silero VAD (v6) through onnxruntime: no PyTorch.

The model is a ~2 MB ONNX file downloaded once from the pinned upstream release and
verified by SHA-256. At 16 kHz it consumes 512-sample (32 ms) frames, plus the last 64
samples of the previous frame as context, mirroring the reference OnnxWrapper.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
FRAME = 512
CONTEXT = 64
MODEL_URL = "https://github.com/snakers4/silero-vad/raw/v6.2.3/src/silero_vad/data/silero_vad.onnx"
MODEL_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
CACHE_DIR = Path(os.environ.get("OPENINTERVIEW_CACHE", Path.home() / ".cache" / "openinterview"))


def model_path() -> Path:
    path = CACHE_DIR / "silero_vad_v6.2.3.onnx"
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as resp:
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        raise RuntimeError(f"Silero VAD checksum mismatch: {digest}")
    tmp.write_bytes(data)
    tmp.rename(path)
    return path


class EnergyVAD:
    """Trivial RMS-threshold VAD. Used in tests and as a fallback if onnxruntime is missing."""

    frame_size = FRAME

    def __init__(self, rms_threshold: float = 0.01) -> None:
        self.rms_threshold = rms_threshold

    def reset(self) -> None:
        pass

    def __call__(self, frame: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(frame))))
        return 1.0 if rms >= self.rms_threshold else 0.0


class SileroVAD:
    """Stateful frame classifier: feed consecutive 512-sample frames, get speech probability."""

    frame_size = FRAME

    def __init__(self, path: Path | None = None) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        self._session = ort.InferenceSession(
            str(path or model_path()), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._input = np.zeros((1, CONTEXT + FRAME), dtype=np.float32)  # reused every call

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape[0] != FRAME:
            raise ValueError(f"expected {FRAME} samples, got {frame.shape[0]}")
        self._input[0, :CONTEXT] = self._context
        self._input[0, CONTEXT:] = frame
        out, self._state = self._session.run(
            None, {"input": self._input, "state": self._state, "sr": self._sr}
        )
        self._context = self._input[0, -CONTEXT:].copy()
        return float(out[0, 0])
