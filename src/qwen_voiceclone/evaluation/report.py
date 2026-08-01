"""Reference-free and optional model-backed checkpoint evaluation."""

from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf

from qwen_voiceclone.data.manifest import read_jsonl


@dataclass(frozen=True)
class Score:
    label: str
    clips: int
    missing: int
    mean_words_per_second: float
    wer: float | None
    speaker_similarity: float | None


def _duration(path: Path) -> float:
    return sf.info(path).duration


def _load_asr():
    try:
        from faster_whisper import WhisperModel
        from jiwer import wer
    except ImportError as exc:
        raise RuntimeError("install evaluation dependencies: pip install -e '.[eval]'") from exc
    print("[eval] loading Whisper large-v3 (first use may download model files)", flush=True)
    return WhisperModel("large-v3", device="cuda", compute_type="float16"), wer


def _load_verifier():
    try:
        import torch
        from speechbrain.inference.speaker import EncoderClassifier
    except ImportError as exc:
        raise RuntimeError("install evaluation dependencies: pip install -e '.[eval]'") from exc
    print("[eval] loading SpeechBrain ECAPA verifier (first use may download model files)", flush=True)
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cuda"}), torch


def score_directory(manifest: Path, audio_dir: Path, reference_audio: Path, label: str, use_models: bool = True) -> Score:
    rows = read_jsonl(manifest)
    durations: list[float] = []
    expected: list[str] = []
    actual: list[str] = []
    similarities: list[float] = []
    missing = 0
    asr = wer_fn = verifier = torch = None
    if use_models:
        asr, wer_fn = _load_asr()
        verifier, torch = _load_verifier()
        reference_embedding = verifier.encode_batch(verifier.load_audio(str(reference_audio))).squeeze()
    for index, row in enumerate(rows, start=1):
        path = audio_dir / f"{row['id']}.wav"
        if not path.exists():
            print(f"[eval] {index}/{len(rows)} missing: {row['id']}", flush=True)
            missing += 1
            continue
        print(f"[eval] scoring {index}/{len(rows)}: {row['id']}", flush=True)
        duration = _duration(path)
        durations.append(len(row["text"].split()) / duration)
        if use_models and asr and wer_fn and verifier and torch:
            segments, _ = asr.transcribe(str(path), beam_size=5)
            actual.append(" ".join(segment.text.strip() for segment in segments))
            expected.append(row["text"])
            embedding = verifier.encode_batch(verifier.load_audio(str(path))).squeeze()
            similarities.append(float(torch.nn.functional.cosine_similarity(reference_embedding, embedding, dim=0)))
    return Score(
        label=label,
        clips=len(rows),
        missing=missing,
        mean_words_per_second=statistics.fmean(durations) if durations else 0.0,
        wer=wer_fn(expected, actual) if use_models and expected and wer_fn else None,
        speaker_similarity=statistics.fmean(similarities) if similarities else None,
    )


def write_score(path: Path, score: Score) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(score), indent=2) + "\n", encoding="utf-8")


def compare(candidate_path: Path, baseline_path: Path, listener_votes: Path | None) -> dict[str, object]:
    candidate = json.loads(candidate_path.read_text())
    baseline = json.loads(baseline_path.read_text())
    votes = 0
    candidate_wins = 0
    if listener_votes:
        with listener_votes.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("winner") not in {"candidate", "baseline"}:
                    raise ValueError("listener vote winner must be candidate or baseline")
                votes += 1
                candidate_wins += row["winner"] == "candidate"
    similarity_ok = (
        candidate["speaker_similarity"] is not None
        and baseline["speaker_similarity"] is not None
        and candidate["speaker_similarity"] >= baseline["speaker_similarity"]
    )
    wer_ok = candidate["wer"] is not None and baseline["wer"] is not None and candidate["wer"] <= baseline["wer"]
    rate_ratio = candidate["mean_words_per_second"] / baseline["mean_words_per_second"] if baseline["mean_words_per_second"] else 0
    rate_ok = 0.85 <= rate_ratio <= 1.15
    votes_ok = votes > 0 and candidate_wins / votes >= 2 / 3
    return {
        "accept": bool(similarity_ok and wer_ok and rate_ok and votes_ok),
        "candidate_vote_rate": candidate_wins / votes if votes else None,
        "similarity_ok": similarity_ok,
        "wer_ok": wer_ok,
        "speaking_rate_ratio": rate_ratio,
        "speaking_rate_ok": rate_ok,
        "votes_ok": votes_ok,
    }
