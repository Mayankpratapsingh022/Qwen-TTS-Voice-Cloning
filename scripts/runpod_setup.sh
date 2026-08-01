#!/usr/bin/env bash
set -Eeuo pipefail

# Run this once inside a RunPod PyTorch pod.  Everything durable lives in /workspace.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
QWEN_REPO_DIR="${QVC_QWEN_REPO_DIR:-/workspace/Qwen3-TTS}"
QWEN_GIT_REF="${QVC_QWEN_GIT_REF:-main}"
LOG_DIR="${PROJECT_DIR}/runs/setup-$(date -u +%Y%m%dT%H%M%SZ)"
export HF_HOME=/workspace/.cache/huggingface
export PIP_CACHE_DIR=/workspace/.cache/pip
export TMPDIR=/workspace/.tmp
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$LOG_DIR" "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
exec > >(tee -a "$LOG_DIR/setup.log") 2>&1

log() { printf '[setup %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
trap 'log "FAILED at line $LINENO"' ERR

log "Checking RunPod GPU and persistent workspace"
test -w /workspace
command -v nvidia-smi
nvidia-smi
python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA is not available"; print("torch=", torch.__version__, "gpu=", torch.cuda.get_device_name(0))'

export PIP_PROGRESS_BAR=on
if [[ ! -d "$QWEN_REPO_DIR/.git" ]]; then
  log "Cloning official Qwen3-TTS with visible git progress"
  git clone --progress https://github.com/QwenLM/Qwen3-TTS.git "$QWEN_REPO_DIR"
fi
git -C "$QWEN_REPO_DIR" fetch --tags --progress origin
git -C "$QWEN_REPO_DIR" checkout "$QWEN_GIT_REF"
git -C "$QWEN_REPO_DIR" rev-parse HEAD | tee "$LOG_DIR/qwen_git_commit.txt"

log "Installing isolated project and official Qwen dependencies with pip progress"
if [[ ! -d "$PROJECT_DIR/.venv" ]]; then
  # Preserve the CUDA-enabled torch supplied by the RunPod PyTorch template.
  python3 -m venv --system-site-packages "$PROJECT_DIR/.venv"
fi
source "$PROJECT_DIR/.venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "$QWEN_REPO_DIR"
python -m pip install -e "$PROJECT_DIR[train,eval,dev]"
log "Installing and checking FlashAttention with visible build/download output"
HOST_MEMORY_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
if [[ "$HOST_MEMORY_KB" -lt 100663296 ]]; then
  export MAX_JOBS=4
  log "Host RAM is below 96GB. Limiting FlashAttention build to MAX_JOBS=$MAX_JOBS"
fi
python -m pip install -U flash-attn --no-build-isolation
python -c 'import flash_attn, torch; assert torch.cuda.is_available(); print("flash_attn=", flash_attn.__version__, "torch_cuda=", torch.version.cuda)'

log "Prefetching public Qwen model weights; Hugging Face progress remains visible"
qwen-voiceclone models prefetch
qwen-voiceclone budget show
log "Setup complete. Inspect $LOG_DIR/setup.log, then run scripts/runpod_train_and_eval.sh"
