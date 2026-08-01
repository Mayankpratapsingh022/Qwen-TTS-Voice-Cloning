#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/qwen3tts-voiceclone}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$PROJECT_DIR/data/processed/mayank}"
RUN_DIR="${RUN_DIR:-$PROJECT_DIR/runs/one-epoch-$(date -u +%Y%m%dT%H%M%SZ)}"
cd "$PROJECT_DIR"
source .venv/bin/activate
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1

log() { printf '[driver %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
trap 'log "FAILED at line $LINENO. Outputs and logs remain in $RUN_DIR"' ERR

test -f "$MANIFEST_ROOT/train.jsonl"
test -f "$MANIFEST_ROOT/holdout.jsonl"
test -f "$MANIFEST_ROOT/reference.txt"
qwen-voiceclone budget show
log "Smoke test: stdout, GPU CSV, W&B metrics, and checkpoints will be live under runs/smoke"
qwen-voiceclone train smoke --manifest "$MANIFEST_ROOT/train.jsonl" --output-dir "$RUN_DIR/smoke"

log "One-epoch fine-tune: watch this terminal, $RUN_DIR/logs, and W&B"
qwen-voiceclone train run --manifest "$MANIFEST_ROOT/train.jsonl" --output-dir "$RUN_DIR/finetune" --epochs 1 --learning-rate 2e-6
CHECKPOINT="$RUN_DIR/finetune/checkpoint-epoch-0"
test -d "$CHECKPOINT"

log "Generating the 20 untouched holdout lines with the zero-shot baseline"
qwen-voiceclone eval generate-baseline --manifest "$MANIFEST_ROOT/holdout.jsonl" --reference-text-file "$MANIFEST_ROOT/reference.txt" --output-dir "$RUN_DIR/baseline_audio"
log "Generating the same holdout lines with the fine-tuned checkpoint"
qwen-voiceclone eval generate-checkpoint --checkpoint "$CHECKPOINT" --manifest "$MANIFEST_ROOT/holdout.jsonl" --output-dir "$RUN_DIR/finetune_audio"

REFERENCE_AUDIO="$(MANIFEST_ROOT="$MANIFEST_ROOT" python -c 'import json, os; from pathlib import Path; print(json.loads((Path(os.environ["MANIFEST_ROOT"]) / "train.jsonl").read_text().splitlines()[0])["ref_audio"])')"
log "Scoring transcript fidelity, speaker similarity, and speaking rate"
qwen-voiceclone eval score --manifest "$MANIFEST_ROOT/holdout.jsonl" --audio-dir "$RUN_DIR/baseline_audio" --reference-audio "$REFERENCE_AUDIO" --label baseline --report "$RUN_DIR/baseline_score.json"
qwen-voiceclone eval score --manifest "$MANIFEST_ROOT/holdout.jsonl" --audio-dir "$RUN_DIR/finetune_audio" --reference-audio "$REFERENCE_AUDIO" --label finetune --report "$RUN_DIR/finetune_score.json"
log "Completed. Add listener votes before qwen-voiceclone eval compare; no model is accepted automatically."
