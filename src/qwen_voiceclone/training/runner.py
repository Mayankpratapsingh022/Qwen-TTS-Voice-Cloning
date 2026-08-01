"""Runs Qwen's official tokenization and SFT scripts with a hard local time limit."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from qwen_voiceclone.config import Settings
from qwen_voiceclone.observability import RunObserver, settings_for_wandb


@dataclass(frozen=True)
class TrainingRequest:
    manifest: Path
    output_dir: Path
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-5
    speaker_name: str = "my_voice"


def _script(settings: Settings, name: str) -> Path:
    path = settings.qwen_repo_dir / "finetuning" / name
    if not path.exists():
        raise FileNotFoundError(f"Qwen script not found: {path}; clone Qwen3-TTS separately and set QVC_QWEN_REPO_DIR")
    return path


def _metadata(request: TrainingRequest, settings: Settings) -> dict[str, object]:
    return {
        "request": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(request).items()},
        "model_id": settings.model_id,
        "tokenizer_id": settings.tokenizer_id,
        "model_revision": settings.model_revision,
        "qwen_repo_dir": str(settings.qwen_repo_dir),
        "gpu_name": settings.gpu_name,
        "gpu_hourly_usd": settings.gpu_hourly_usd,
        "budget_usd": settings.budget_usd,
        "allowed_gpu_hours": settings.allowed_gpu_hours,
        "created_at": int(time.time()),
    }


def _run(command: list[str], timeout_seconds: int, cwd: Path, dry_run: bool, observer: RunObserver, stage: str) -> None:
    observer.command(stage, command)
    if dry_run:
        observer.event("dry_run", {"stage": stage})
        return
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        for line in process.stdout:
            lines.put(line.rstrip("\n"))
        lines.put(None)

    reader = threading.Thread(target=read_output, name=f"qvc-{stage}-stdout", daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout_seconds
    try:
        while True:
            if time.monotonic() >= deadline:
                process.kill()
                raise RuntimeError("training stopped by the configured GPU budget guard")
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                if process.poll() is not None:
                    break
                continue
            if line is None:
                break
            observer.line(stage, line)
        return_code = process.wait(timeout=10)
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        if process.poll() is None:
            process.kill()


def tokenize(request: TrainingRequest, settings: Settings, timeout_seconds: int, observer: RunObserver, dry_run: bool = False) -> Path:
    request.output_dir.mkdir(parents=True, exist_ok=True)
    output = request.output_dir / "train_with_codes.jsonl"
    command = [
        sys.executable,
        str(_script(settings, "prepare_data.py")),
        "--device",
        "cuda:0",
        "--tokenizer_model_path",
        settings.tokenizer_id,
        "--input_jsonl",
        str(request.manifest.resolve()),
        "--output_jsonl",
        str(output.resolve()),
    ]
    _run(command, timeout_seconds, settings.qwen_repo_dir, dry_run, observer, "tokenize")
    return output


def _smoke_manifest(manifest: Path, output_dir: Path) -> Path:
    """Use a small deterministic subset so the preflight cannot consume the full budget."""
    rows = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("training manifest is empty")
    smoke_path = output_dir / "smoke_input.jsonl"
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    smoke_path.write_text("\n".join(rows[: min(8, len(rows))]) + "\n", encoding="utf-8")
    return smoke_path


def train(request: TrainingRequest, settings: Settings, smoke: bool = False, dry_run: bool = False) -> Path:
    if request.epochs < 1 or request.batch_size < 1:
        raise ValueError("epochs and batch_size must be at least 1")
    request.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(request, settings)
    (request.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    allowed_seconds = int((settings.smoke_gpu_hours if smoke else settings.full_gpu_hours) * 3600)
    deadline = time.monotonic() + allowed_seconds
    observer = RunObserver(request.output_dir, settings, {**settings_for_wandb(settings), **metadata})
    observer.start()
    try:
        active_request = replace(request, manifest=_smoke_manifest(request.manifest, request.output_dir)) if smoke else request
        tokenized = tokenize(active_request, settings, timeout_seconds=max(1, int(deadline - time.monotonic())), observer=observer, dry_run=dry_run)
        command = [
            sys.executable,
            str(_script(settings, "sft_12hz.py")),
            "--init_model_path",
            settings.model_id,
            "--output_model_path",
            str(active_request.output_dir.resolve()),
            "--train_jsonl",
            str(tokenized.resolve()),
            "--batch_size",
            str(1 if smoke else request.batch_size),
            "--lr",
            str(request.learning_rate),
            "--num_epochs",
            str(1 if smoke else request.epochs),
            "--speaker_name",
            request.speaker_name,
        ]
        _run(command, max(1, int(deadline - time.monotonic())), settings.qwen_repo_dir, dry_run, observer, "sft")
    except Exception:
        observer.finish("failed")
        raise
    observer.finish("success")
    return request.output_dir
