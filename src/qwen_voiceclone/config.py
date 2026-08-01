"""Runtime configuration for the isolated Qwen project."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="QVC_", extra="ignore")

    speaker_id: str = "my_voice"
    model_id: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    tokenizer_id: str = "Qwen/Qwen3-TTS-Tokenizer-12Hz"
    model_revision: str = "main"
    qwen_repo_dir: Path = Path("/workspace/Qwen3-TTS")
    workspace_dir: Path = Path("/workspace/qwen3tts-voiceclone")
    gpu_name: str = "A100-80GB"
    gpu_hourly_usd: float = 1.39
    budget_usd: float = 10.0
    max_gpu_hours: float = 6.0
    smoke_gpu_hours: float = 0.5
    full_gpu_hours: float = 4.5
    evaluation_gpu_hours: float = 1.0
    min_clip_seconds: float = 1.0
    max_clip_seconds: float = 30.0
    train_fraction: float = 0.90
    validation_fraction: float = 0.05
    holdout_fraction: float = 0.05
    wandb_mode: Literal["online", "offline", "disabled"] = "online"
    wandb_project: str = "qwen3tts-voiceclone"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    gpu_poll_seconds: int = 20

    @field_validator(
        "gpu_hourly_usd", "budget_usd", "max_gpu_hours", "smoke_gpu_hours", "full_gpu_hours", "evaluation_gpu_hours", "min_clip_seconds", "max_clip_seconds", "gpu_poll_seconds"
    )
    @classmethod
    def positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @model_validator(mode="after")
    def valid_budget_partition(self) -> Settings:
        if self.smoke_gpu_hours + self.full_gpu_hours + self.evaluation_gpu_hours > self.allowed_gpu_hours:
            raise ValueError("smoke_gpu_hours + full_gpu_hours + evaluation_gpu_hours cannot exceed the configured GPU budget")
        return self

    @property
    def allowed_gpu_hours(self) -> float:
        return min(self.max_gpu_hours, self.budget_usd / self.gpu_hourly_usd)

    @property
    def training_timeout_seconds(self) -> int:
        return int(self.allowed_gpu_hours * 3600)


def get_settings() -> Settings:
    return Settings()
