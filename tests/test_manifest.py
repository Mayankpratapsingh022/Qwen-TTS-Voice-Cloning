from pathlib import Path

import soundfile as sf

from qwen_voiceclone.config import Settings
from qwen_voiceclone.data.huggingface import convert_kaldi_dataset
from qwen_voiceclone.data.manifest import read_jsonl
from qwen_voiceclone.data.pipeline import prepare_dataset


def _write_wav(path: Path) -> None:
    sf.write(path, [0.0] * 16000, 16000)


def test_prepare_creates_qwen_manifests(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    lines = ["audio,text"]
    for session in ("session_a", "session_b", "session_c"):
        (raw / session).mkdir(parents=True)
        for index in range(4):
            relative = f"{session}/{index}.wav"
            _write_wav(raw / relative)
            lines.append(f"{relative},Example sentence {session} {index}")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("\n".join(lines) + "\n")
    reference = tmp_path / "reference.wav"
    _write_wav(reference)
    dest = tmp_path / "processed"

    result = prepare_dataset(raw, metadata, reference, dest, Settings())

    assert result.train_count + result.validation_count + result.holdout_count == 12
    assert not result.clip_level_fallback
    row = read_jsonl(dest / "train.jsonl")[0]
    assert set(row) == {"id", "audio", "text", "ref_audio"}
    assert row["ref_audio"] == str(reference.resolve())


def test_convert_kaldi_dataset_preserves_source_splits(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    chunks = dataset / "chunks"
    chunks.mkdir(parents=True)
    for split, ids in {"train": ["train_a", "train_b"], "cv": ["cv_a"], "holdout": ["holdout_a"]}.items():
        lines_text: list[str] = []
        lines_audio: list[str] = []
        (dataset / split).mkdir()
        for utterance in ids:
            wav = chunks / f"{utterance}.wav"
            _write_wav(wav)
            lines_text.append(f"{utterance} Example transcript for {utterance}")
            lines_audio.append(f"{utterance} /workspace/dataset/mayank/chunks/{utterance}.wav")
        (dataset / split / "text").write_text("\n".join(lines_text) + "\n")
        (dataset / split / "wav.scp").write_text("\n".join(lines_audio) + "\n")

    dest = tmp_path / "processed"
    result = convert_kaldi_dataset(dataset, dest, Settings())

    assert (result.train_count, result.validation_count, result.holdout_count) == (2, 1, 1)
    assert result.skipped_count == 0
    assert read_jsonl(dest / "validation.jsonl")[0]["id"] == "cv_a"
    assert (dest / "reference.txt").read_text().strip() == "Example transcript for train_a"
