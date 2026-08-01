"""Strict JSONL manifests for Qwen's official fine-tuning scripts."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Clip:
    id: str
    audio: str
    text: str
    session: str
    seconds: float

    def qwen_row(self, reference_audio: Path) -> dict[str, str]:
        return {"id": self.id, "audio": self.audio, "text": self.text, "ref_audio": str(reference_audio)}


def read_metadata(raw_dir: Path, metadata_path: Path) -> list[tuple[Path, str]]:
    """Read the documented `audio,text` CSV with safe path resolution."""
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) < {"audio", "text"}:
        raise ValueError("metadata must be a non-empty CSV with audio,text headers")
    items: list[tuple[Path, str]] = []
    for line, row in enumerate(rows, start=2):
        audio = Path(row["audio"]).expanduser()
        path = audio if audio.is_absolute() else raw_dir / audio
        if not row["text"].strip():
            raise ValueError(f"metadata line {line} has empty text")
        items.append((path.resolve(), row["text"].strip()))
    return items


def write_jsonl(path: Path, clips: list[Clip], reference_audio: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for clip in clips:
            handle.write(json.dumps(clip.qwen_row(reference_audio), ensure_ascii=False) + "\n")


def write_inventory(path: Path, clips: list[Clip]) -> None:
    path.write_text(json.dumps([asdict(clip) for clip in clips], indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
