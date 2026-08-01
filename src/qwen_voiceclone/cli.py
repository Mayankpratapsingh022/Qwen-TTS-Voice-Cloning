"""Command-line interface for the self-contained Qwen project."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print

from qwen_voiceclone.config import get_settings
from qwen_voiceclone.data.huggingface import import_huggingface_dataset, prefetch_qwen_models
from qwen_voiceclone.data.pipeline import prepare_dataset
from qwen_voiceclone.evaluation.report import compare, score_directory, write_score
from qwen_voiceclone.inference.generate import generate_custom_voice, generate_holdout, generate_zero_shot
from qwen_voiceclone.training.runner import TrainingRequest, train

app = typer.Typer(no_args_is_help=True)
data_app = typer.Typer(no_args_is_help=True)
train_app = typer.Typer(no_args_is_help=True)
eval_app = typer.Typer(no_args_is_help=True)
budget_app = typer.Typer(no_args_is_help=True)
models_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")
app.add_typer(train_app, name="train")
app.add_typer(eval_app, name="eval")
app.add_typer(budget_app, name="budget")
app.add_typer(models_app, name="models")


@data_app.command("prepare")
def data_prepare(
    raw_dir: Path = typer.Option(..., exists=True, file_okay=False),
    metadata: Path = typer.Option(..., exists=True, dir_okay=False),
    reference_audio: Path = typer.Option(..., exists=True, dir_okay=False),
    dest: Path = typer.Option(...),
) -> None:
    """Validate clips and create Qwen-compatible JSONL splits."""
    result = prepare_dataset(raw_dir, metadata, reference_audio, dest, get_settings())
    print({"train": result.train_count, "validation": result.validation_count, "holdout": result.holdout_count,
           "sessions": result.session_count, "clip_level_fallback": result.clip_level_fallback})


@data_app.command("import-hf")
def data_import_hf(
    repo_id: str = typer.Option(..., help="Hugging Face dataset repository, e.g. Mayank022/voiceclone-mayank-dataset"),
    download_dir: Path = typer.Option(Path("data/huggingface"), file_okay=False),
    dest: Path = typer.Option(Path("data/processed/mayank"), file_okay=False),
    revision: str = typer.Option("main"),
    reference_utterance: str | None = typer.Option(None, help="Optional train utterance ID for zero-shot reference"),
) -> None:
    """Download a private Kaldi dataset and preserve its train/cv/holdout splits for Qwen."""
    result = import_huggingface_dataset(
        repo_id, download_dir, dest, get_settings(), revision=revision, reference_utterance=reference_utterance
    )
    print(
        {
            "dataset_dir": str(result.dataset_dir),
            "train": result.train_count,
            "validation": result.validation_count,
            "holdout": result.holdout_count,
            "skipped": result.skipped_count,
            "reference_audio": str(result.reference_audio),
            "reference_text_file": str(dest / "reference.txt"),
        }
    )


def _request(manifest: Path, output_dir: Path, epochs: int, batch_size: int, learning_rate: float, speaker: str) -> TrainingRequest:
    return TrainingRequest(manifest=manifest, output_dir=output_dir, epochs=epochs, batch_size=batch_size,
                           learning_rate=learning_rate, speaker_name=speaker)


@train_app.command("smoke")
def train_smoke(
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(Path("runs/smoke")),
    speaker: str = typer.Option(None),
    dry_run: bool = typer.Option(False),
) -> None:
    """Run data coding plus one low-batch epoch to verify the A100 environment."""
    settings = get_settings()
    train(_request(manifest, output_dir, 1, 1, 2e-5, speaker or settings.speaker_id), settings, smoke=True, dry_run=dry_run)


@train_app.command("run")
def train_run(
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(...),
    epochs: int = typer.Option(3, min=1, max=3),
    batch_size: int = typer.Option(2, min=1),
    learning_rate: float = typer.Option(2e-6, min=1e-8),
    speaker: str = typer.Option(None),
    dry_run: bool = typer.Option(False),
) -> None:
    """Run the budget-capped Qwen SFT (start with one epoch)."""
    settings = get_settings()
    train(_request(manifest, output_dir, epochs, batch_size, learning_rate, speaker or settings.speaker_id), settings, dry_run=dry_run)


@app.command("generate")
def generate(
    checkpoint: Path = typer.Option(..., exists=True, file_okay=False),
    text: str = typer.Option(...),
    output: Path = typer.Option(...),
    speaker: str = typer.Option(None),
) -> None:
    """Generate one WAV from the selected fine-tuned checkpoint."""
    generate_custom_voice(checkpoint, text, speaker or get_settings().speaker_id, output)


@app.command("generate-zero-shot")
def generate_zero_shot_command(
    text: str = typer.Option(...),
    reference_audio: Path = typer.Option(..., exists=True, dir_okay=False),
    reference_text: str = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    """Generate the reference-conditioned Qwen Base-model baseline."""
    generate_zero_shot(get_settings().model_id, text, reference_audio, reference_text, output)


@eval_app.command("score")
def eval_score(
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    audio_dir: Path = typer.Option(..., exists=True, file_okay=False),
    reference_audio: Path = typer.Option(..., exists=True, dir_okay=False),
    label: str = typer.Option(...),
    report: Path = typer.Option(...),
    quick: bool = typer.Option(False, help="Skip ASR and speaker model; duration/rate only."),
) -> None:
    """Score holdout outputs; output files must be named <manifest id>.wav."""
    score = score_directory(manifest, audio_dir, reference_audio, label, use_models=not quick)
    write_score(report, score)
    print(score)


@eval_app.command("compare")
def eval_compare(
    candidate: Path = typer.Option(..., exists=True, dir_okay=False),
    baseline: Path = typer.Option(..., exists=True, dir_okay=False),
    listener_votes: Path | None = typer.Option(None, exists=True, dir_okay=False),
) -> None:
    """Apply the objective + listener acceptance rule."""
    print(json.dumps(compare(candidate, baseline, listener_votes), indent=2))


@budget_app.command("show")
def budget_show() -> None:
    """Show the local budget guard; verify provider pricing before renting."""
    settings = get_settings()
    print({"gpu": settings.gpu_name, "hourly_usd": settings.gpu_hourly_usd, "budget_usd": settings.budget_usd,
           "configured_max_hours": settings.max_gpu_hours, "allowed_hours": settings.allowed_gpu_hours,
           "smoke_hours": settings.smoke_gpu_hours, "full_hours": settings.full_gpu_hours,
           "evaluation_hours": settings.evaluation_gpu_hours,
           "maximum_compute_usd": round(settings.allowed_gpu_hours * settings.gpu_hourly_usd, 2)})


@models_app.command("prefetch")
def models_prefetch(cache_dir: Path | None = typer.Option(None, file_okay=False)) -> None:
    """Download Qwen weights before training so all transfer progress is visible."""
    settings = get_settings()
    model_path, tokenizer_path = prefetch_qwen_models(settings.model_id, settings.tokenizer_id, cache_dir)
    print({"model_cache": str(model_path), "tokenizer_cache": str(tokenizer_path)})


@eval_app.command("generate-baseline")
def eval_generate_baseline(
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    reference_text_file: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(...),
    language: str = typer.Option("English"),
) -> None:
    """Generate every holdout sentence with the Qwen Base zero-shot clone."""
    generate_holdout(get_settings().model_id, manifest, reference_text_file.read_text(encoding="utf-8").strip(), output_dir, "baseline", language)


@eval_app.command("generate-checkpoint")
def eval_generate_checkpoint(
    checkpoint: Path = typer.Option(..., exists=True, file_okay=False),
    manifest: Path = typer.Option(..., exists=True, dir_okay=False),
    output_dir: Path = typer.Option(...),
    speaker: str = typer.Option(None),
    language: str = typer.Option("English"),
) -> None:
    """Generate every holdout sentence from one fine-tuned checkpoint."""
    generate_holdout(str(checkpoint), manifest, None, output_dir, "checkpoint", language, speaker or get_settings().speaker_id)
