#!/usr/bin/env bash
set -euo pipefail

# Run only inside the new qwen3tts-voiceclone checkout on an A100 machine.
# Configure the provider's auto-stop separately; this script never accesses CosyVoice.
PROJECT_DIR="${PROJECT_DIR:-/workspace/qwen3tts-voiceclone}"
cd "$PROJECT_DIR"
if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

qwen-voiceclone budget show
qwen-voiceclone train smoke --manifest "${MANIFEST:-data/processed/mayank/train.jsonl}"
qwen-voiceclone train run --manifest "${MANIFEST:-data/processed/mayank/train.jsonl}" --output-dir "${OUTPUT_DIR:-runs/full}" --epochs 1
