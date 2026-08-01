from qwen_voiceclone.observability import parse_training_metrics


def test_parses_official_qwen_loss_line() -> None:
    assert parse_training_metrics("Epoch 2 | Step 10 | Loss: 0.1234") == {"epoch": 2.0, "step": 10.0, "loss": 0.1234}


def test_ignores_non_metric_line() -> None:
    assert parse_training_metrics("Downloading tokenizer files") is None
