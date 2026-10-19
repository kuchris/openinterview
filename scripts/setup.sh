#!/usr/bin/env bash
# One-shot setup: checks the machine, installs Python deps with uv, builds the system-audio
# helper, pre-downloads the default model, and runs the environment check.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "This project requires macOS on Apple Silicon (MLX runs on Metal)." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "note: ffmpeg not found. It is only needed for 'transcribe-file' and the benchmark."
  echo "      Install with: brew install ffmpeg"
fi

uv sync

if command -v swiftc >/dev/null 2>&1; then
  scripts/build_helper.sh
else
  echo "note: swiftc not found; system audio needs Xcode Command Line Tools:"
  echo "      xcode-select --install"
fi

echo "Downloading default model (small.en) and Silero VAD..."
uv run python -c "
from openinterview.asr.vad import model_path
from openinterview.asr.whisper_engine import WhisperEngine
model_path()
WhisperEngine('small.en').load()
print('models ready')"

uv run openinterview check || true
