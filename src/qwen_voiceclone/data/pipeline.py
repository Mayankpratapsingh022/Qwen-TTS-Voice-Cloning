"""Audio validation and deterministic session-aware splitting."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from qwen_voiceclone.config import Settings
from qwen_voiceclone.data.manifest import Clip, read_metadata, write_inventory, write_jsonl


@dataclass(frozen=True)
class PreparationResult:
    train_count: int
    validation_count: int
    holdout_count: int
    session_count: int
    clip_level_fallback: bool


def _clip_id(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:10]
    return f"{path.stem}-{digest}"


def _duration(path: Path) -> tuple[float, int, int]:
    info = sf.info(path)
    return info.duration, info.channels, info.samplerate


def _validate(path: Path, settings: Settings) -> tuple[float, int]:
    if not path.exists() or not path.is_file():
        raise ValueError(f"audio file does not exist: {path}")
    try:
        seconds, channels, sample_rate = _duration(path)
    except RuntimeError as exc:
        raise ValueError(f"cannot read audio file {path}: {exc}") from exc
    if channels != 1:
        raise ValueError(f"audio must be mono ({path} has {channels} channels)")
    if not settings.min_clip_seconds <= seconds <= settings.max_clip_seconds:
        raise ValueError(
            f"audio must be {settings.min_clip_seconds:g}-{settings.max_clip_seconds:g}s ({path} is {seconds:.2f}s)"
        )
    if sample_rate < 16000:
        raise ValueError(f"audio sample rate must be at least 16kHz ({path} is {sample_rate}Hz)")
    return seconds, sample_rate


def _session_for(path: Path, raw_dir: Path) -> str:
    relative = path.relative_to(raw_dir)
    return relative.parts[0] if len(relative.parts) > 1 else "__root__"


def _split(clips: list[Clip], settings: Settings) -> tuple[dict[str, list[Clip]], bool]:
    by_session: dict[str, list[Clip]] = defaultdict(list)
    for clip in clips:
        by_session[clip.session].append(clip)
    names = sorted(by_session)
    rng = random.Random(23)
    rng.shuffle(names)

    # Preserve whole sessions when possible. Tiny datasets cannot fill all three splits this way.
    if len(names) >= 3:
        # Always keep two complete sessions out of training. With only three sessions this
        # necessarily sacrifices the requested 90/5/5 ratio, but preserves the stronger
        # guarantee: no audio session appears in more than one split.
        splits: dict[str, list[Clip]] = {"train": [], "validation": [], "holdout": []}
        for name in names[:-2]:
            splits["train"].extend(by_session[name])
        splits["validation"].extend(by_session[names[-2]])
        splits["holdout"].extend(by_session[names[-1]])
        return splits, False
    return _split_clip_level(clips), True


def _split_clip_level(clips: list[Clip]) -> dict[str, list[Clip]]:
    ordered = sorted(clips, key=lambda clip: clip.id)
    rng = random.Random(23)
    rng.shuffle(ordered)
    count = len(ordered)
    validation = max(1, round(count * 0.05))
    holdout = max(1, round(count * 0.05))
    if count < 12:
        raise ValueError("need at least 12 clips to make non-empty train, validation, and holdout splits")
    return {
        "holdout": ordered[:holdout],
        "validation": ordered[holdout : holdout + validation],
        "train": ordered[holdout + validation :],
    }


def prepare_dataset(raw_dir: Path, metadata_path: Path, reference_audio: Path, dest: Path, settings: Settings) -> PreparationResult:
    raw_dir = raw_dir.resolve()
    reference_audio = reference_audio.resolve()
    _validate(reference_audio, settings)
    clips: list[Clip] = []
    seen_paths: set[Path] = set()
    for path, text in read_metadata(raw_dir, metadata_path):
        if path in seen_paths:
            raise ValueError(f"duplicate audio entry: {path}")
        seen_paths.add(path)
        seconds, _ = _validate(path, settings)
        try:
            session = _session_for(path, raw_dir)
        except ValueError as exc:
            raise ValueError(f"audio must be inside --raw-dir: {path}") from exc
        clips.append(Clip(id=_clip_id(path), audio=str(path), text=text, session=session, seconds=seconds))
    if len(clips) < 12:
        raise ValueError("need at least 12 validated clips")
    splits, clip_level_fallback = _split(clips, settings)
    dest.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        write_jsonl(dest / f"{name}.jsonl", split, reference_audio)
    write_inventory(dest / "inventory.json", clips)
    (dest / "README.txt").write_text(
        "Do not edit manifests after tokenization. "
        f"session_count={len({clip.session for clip in clips})}; clip_level_fallback={clip_level_fallback}\n",
        encoding="utf-8",
    )
    return PreparationResult(
        train_count=len(splits["train"]),
        validation_count=len(splits["validation"]),
        holdout_count=len(splits["holdout"]),
        session_count=len({clip.session for clip in clips}),
        clip_level_fallback=clip_level_fallback,
    )
