"""Import a private Kaldi-style Hugging Face dataset into Qwen JSONL manifests."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

from qwen_voiceclone.config import Settings
from qwen_voiceclone.data.manifest import Clip, write_inventory, write_jsonl
from qwen_voiceclone.data.pipeline import _validate

_SPLIT_NAMES = {"train": "train", "cv": "validation", "holdout": "holdout"}


@dataclass(frozen=True)
class HuggingFaceImportResult:
    dataset_dir: Path
    train_count: int
    validation_count: int
    holdout_count: int
    skipped_count: int
    reference_audio: Path
    reference_text: str


def _token() -> str | None:
    """Read the token without ever writing or displaying it."""
    return os.environ.get("HF_TOKEN") or dotenv_values(".env").get("HF_TOKEN")


def download_dataset(repo_id: str, local_dir: Path, revision: str = "main") -> Path:
    """Download only the source splits and their WAV chunks from a dataset repository."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('Install the project dependencies first: pip install -e ".[dev]"') from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=local_dir,
        token=_token(),
        allow_patterns=["chunks/*.wav", "train/*", "cv/*", "holdout/*"],
    )
    return local_dir.resolve()


def prefetch_qwen_models(model_id: str, tokenizer_id: str, cache_dir: Path | None = None) -> tuple[Path, Path]:
    """Warm the persistent Hugging Face cache with progress bars before a costly run."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError('Install the project dependencies first: pip install -e ".[dev]"') from exc
    kwargs = {"cache_dir": str(cache_dir)} if cache_dir else {}
    print(f"[download] model: {model_id}", flush=True)
    model_path = Path(snapshot_download(repo_id=model_id, **kwargs))
    print(f"[download] tokenizer: {tokenizer_id}", flush=True)
    tokenizer_path = Path(snapshot_download(repo_id=tokenizer_id, **kwargs))
    return model_path, tokenizer_path


def _read_key_value(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"required Kaldi manifest is missing: {path}")
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        utterance, separator, value = line.partition(" ")
        if not separator or not utterance or not value.strip():
            raise ValueError(f"invalid Kaldi manifest entry at {path}:{number}")
        if utterance in values:
            raise ValueError(f"duplicate utterance {utterance!r} in {path}")
        values[utterance] = value.strip()
    return values


def _audio_path(dataset_dir: Path, source_path: str) -> Path:
    """Map the prior CosyVoice absolute path to this isolated dataset download."""
    name = Path(source_path).name
    path = dataset_dir / "chunks" / name
    if not path.is_file():
        raise ValueError(f"audio referenced by the manifest was not downloaded: {name}")
    return path.resolve()


def _read_split(dataset_dir: Path, split: str, settings: Settings) -> tuple[list[Clip], list[str]]:
    text = _read_key_value(dataset_dir / split / "text")
    audio = _read_key_value(dataset_dir / split / "wav.scp")
    clips: list[Clip] = []
    skipped: list[str] = []
    for utterance, transcript in text.items():
        source_path = audio.get(utterance)
        if source_path is None:
            skipped.append(f"{utterance}\tmissing from wav.scp")
            continue
        path = _audio_path(dataset_dir, source_path)
        try:
            seconds, _ = _validate(path, settings)
        except ValueError as exc:
            skipped.append(f"{utterance}\t{exc}")
            continue
        clips.append(Clip(id=utterance, audio=str(path), text=transcript, session=split, seconds=seconds))
    orphaned_audio = sorted(set(audio) - set(text))
    skipped.extend(f"{utterance}\tmissing from text" for utterance in orphaned_audio)
    if not clips:
        raise ValueError(f"no usable clips in {split}")
    return clips, skipped


def _select_reference(clips: list[Clip], requested_utterance: str | None) -> Clip:
    if requested_utterance:
        for clip in clips:
            if clip.id == requested_utterance:
                return clip
        raise ValueError(f"reference utterance is not a usable train clip: {requested_utterance}")
    # A 6-12 second clean excerpt gives Qwen enough vocal context without excessive prompt length.
    candidates = [clip for clip in clips if 6 <= clip.seconds <= 12]
    return max(candidates or clips, key=lambda clip: clip.seconds)


def convert_kaldi_dataset(
    dataset_dir: Path,
    dest: Path,
    settings: Settings,
    reference_utterance: str | None = None,
) -> HuggingFaceImportResult:
    """Validate existing train/cv/holdout splits and write Qwen's JSONL contract."""
    dataset_dir = dataset_dir.resolve()
    splits: dict[str, list[Clip]] = {}
    skipped: list[str] = []
    for source_name, qwen_name in _SPLIT_NAMES.items():
        clips, rejected = _read_split(dataset_dir, source_name, settings)
        splits[qwen_name] = clips
        skipped.extend(f"{source_name}\t{reason}" for reason in rejected)

    reference = _select_reference(splits["train"], reference_utterance)
    dest.mkdir(parents=True, exist_ok=True)
    for name, clips in splits.items():
        write_jsonl(dest / f"{name}.jsonl", clips, Path(reference.audio))
    write_inventory(dest / "inventory.json", [clip for clips in splits.values() for clip in clips])
    (dest / "reference.txt").write_text(reference.text + "\n", encoding="utf-8")
    (dest / "reference_utterance.txt").write_text(reference.id + "\n", encoding="utf-8")
    (dest / "skipped.tsv").write_text("\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8")
    (dest / "README.txt").write_text(
        "Imported from a Hugging Face Kaldi-style dataset. Existing train/cv/holdout boundaries were preserved.\n"
        "reference.txt is the exact transcript for the selected zero-shot reference clip.\n",
        encoding="utf-8",
    )
    return HuggingFaceImportResult(
        dataset_dir=dataset_dir,
        train_count=len(splits["train"]),
        validation_count=len(splits["validation"]),
        holdout_count=len(splits["holdout"]),
        skipped_count=len(skipped),
        reference_audio=Path(reference.audio),
        reference_text=reference.text,
    )


def import_huggingface_dataset(
    repo_id: str,
    download_dir: Path,
    dest: Path,
    settings: Settings,
    revision: str = "main",
    reference_utterance: str | None = None,
) -> HuggingFaceImportResult:
    dataset_dir = download_dataset(repo_id, download_dir, revision)
    return convert_kaldi_dataset(dataset_dir, dest, settings, reference_utterance)
