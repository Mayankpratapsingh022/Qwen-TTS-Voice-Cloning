import json
from pathlib import Path

from qwen_voiceclone.evaluation.report import compare


def test_compare_accepts_better_candidate_with_votes(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    candidate.write_text(json.dumps({"speaker_similarity": 0.9, "wer": 0.05, "mean_words_per_second": 2.1}))
    baseline.write_text(json.dumps({"speaker_similarity": 0.8, "wer": 0.06, "mean_words_per_second": 2.0}))
    votes = tmp_path / "votes.csv"
    votes.write_text("winner\ncandidate\ncandidate\nbaseline\n")

    result = compare(candidate, baseline, votes)

    assert result["accept"]
