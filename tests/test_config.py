from qwen_voiceclone.config import Settings


def test_budget_guard_uses_lower_limit() -> None:
    settings = Settings(gpu_hourly_usd=2.0, budget_usd=20.0, max_gpu_hours=12.5, smoke_gpu_hours=0.5, full_gpu_hours=8.5, evaluation_gpu_hours=1.0)
    assert settings.allowed_gpu_hours == 10.0
    assert settings.training_timeout_seconds == 36000
