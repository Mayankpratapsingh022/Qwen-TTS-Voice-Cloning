"""Visible, durable training telemetry for an ephemeral GPU pod."""

from __future__ import annotations

import csv
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from qwen_voiceclone.config import Settings

_LOSS = re.compile(r"Epoch\s+(?P<epoch>\d+)\s+\|\s+Step\s+(?P<step>\d+)\s+\|\s+Loss:\s+(?P<loss>[\d.]+)")


def parse_training_metrics(line: str) -> dict[str, float] | None:
    """Extract the stable log line emitted by Qwen's official SFT script."""
    match = _LOSS.search(line)
    if not match:
        return None
    return {key: float(value) for key, value in match.groupdict().items()}


class RunObserver:
    """Tee subprocess output, poll GPU health, and optionally send small metrics to W&B."""

    def __init__(self, output_dir: Path, settings: Settings, config: dict[str, Any]) -> None:
        self.output_dir = output_dir
        self.settings = settings
        self.config = config
        self.logs_dir = output_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._wandb: Any | None = None
        self._wandb_run: Any | None = None
        self._gpu_stop = threading.Event()
        self._gpu_thread: threading.Thread | None = None
        self._gpu_csv: Any | None = None
        self._gpu_writer: Any | None = None
        self._loss_step = 0

    def start(self) -> None:
        self._start_wandb()
        self._start_gpu_monitor()
        self.event("run_started", {"pid": os.getpid()})

    def finish(self, status: str) -> None:
        self.event("run_finished", {"status": status})
        self._gpu_stop.set()
        if self._gpu_thread:
            self._gpu_thread.join(timeout=self.settings.gpu_poll_seconds + 2)
        if self._gpu_csv:
            self._gpu_csv.close()
        if self._wandb_run:
            self._wandb_run.finish(exit_code=0 if status == "success" else 1)

    def event(self, name: str, values: dict[str, Any]) -> None:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload = {"timestamp": timestamp, "event": name, **values}
        print(f"[qvc] {payload}", flush=True)
        with (self.logs_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            import json

            handle.write(json.dumps(payload) + "\n")
        if self._wandb_run:
            self._wandb_run.log({f"event/{name}": 1, **{f"event/{key}": value for key, value in values.items()}})

    def command(self, stage: str, command: list[str]) -> None:
        rendered = " ".join(command)
        print(f"[qvc:{stage}] $ {rendered}", flush=True)
        with (self.logs_dir / "commands.log").open("a", encoding="utf-8") as handle:
            handle.write(f"{stage}: {rendered}\n")

    def line(self, stage: str, line: str) -> None:
        print(f"[{stage}] {line}")
        with (self.logs_dir / f"{stage}.log").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        metrics = parse_training_metrics(line)
        if metrics and self._wandb_run:
            self._loss_step += 1
            self._wandb_run.log({"train/loss": metrics["loss"], "train/epoch": metrics["epoch"], "train/step": metrics["step"]}, step=self._loss_step)

    def _start_wandb(self) -> None:
        if self.settings.wandb_mode == "disabled":
            return
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError('Install training dependencies: pip install -e ".[train]"') from exc
        if self.settings.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is required for online W&B logging; use a RunPod Secret or set QVC_WANDB_MODE=offline")
        wandb_dir = self.output_dir / "wandb"
        for key, path in {
            "WANDB_DIR": wandb_dir,
            "WANDB_CACHE_DIR": wandb_dir / "cache",
            "WANDB_DATA_DIR": wandb_dir / "data",
            "WANDB_CONFIG_DIR": wandb_dir / "config",
        }.items():
            path.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault(key, str(path))
        self._wandb = wandb
        try:
            self._wandb_run = wandb.init(
                project=self.settings.wandb_project,
                entity=self.settings.wandb_entity,
                name=self.settings.wandb_run_name,
                mode=self.settings.wandb_mode,
                dir=str(wandb_dir),
                config=self.config,
                tags=["qwen3-tts", "voice-clone", self.settings.gpu_name],
            )
        except Exception as exc:  # noqa: BLE001 - third-party telemetry must never stop checkpointing.
            # Telemetry must never discard an expensive checkpoint; terminal and CSV logs continue.
            print(f"[qvc] W&B unavailable ({exc}); continuing with durable local logs", flush=True)
            self._wandb_run = None
            return
        print(f"[qvc] W&B mode={self.settings.wandb_mode}; run={self._wandb_run.url or 'local/offline'}", flush=True)

    def _start_gpu_monitor(self) -> None:
        self._gpu_csv = (self.logs_dir / "gpu.csv").open("w", newline="", encoding="utf-8")
        self._gpu_writer = csv.DictWriter(
            self._gpu_csv,
            fieldnames=["timestamp", "index", "name", "temperature_c", "utilization_pct", "memory_used_mib", "memory_total_mib", "power_w"],
        )
        self._gpu_writer.writeheader()
        self._gpu_thread = threading.Thread(target=self._poll_gpu, name="qvc-gpu-monitor", daemon=True)
        self._gpu_thread.start()

    def _poll_gpu(self) -> None:
        query = "timestamp,index,name,temperature.gpu,utilization.gpu,memory.used,memory.total,power.draw"
        while not self._gpu_stop.is_set():
            try:
                completed = subprocess.run(
                    ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
                for raw in completed.stdout.splitlines():
                    parts = [part.strip() for part in raw.split(",")]
                    if len(parts) != 8:
                        continue
                    row = dict(zip(self._gpu_writer.fieldnames, parts))
                    self._gpu_writer.writerow(row)
                    self._gpu_csv.flush()
                    print(
                        f"[gpu] {row['name']} util={row['utilization_pct']}% "
                        f"vram={row['memory_used_mib']}/{row['memory_total_mib']}MiB temp={row['temperature_c']}C",
                        flush=True,
                    )
                    if self._wandb_run:
                        self._wandb_run.log(
                            {
                                "gpu/utilization_pct": float(row["utilization_pct"]),
                                "gpu/memory_used_mib": float(row["memory_used_mib"]),
                                "gpu/memory_total_mib": float(row["memory_total_mib"]),
                                "gpu/temperature_c": float(row["temperature_c"]),
                                "gpu/power_w": float(row["power_w"]),
                            }
                        )
            except (FileNotFoundError, subprocess.SubprocessError) as exc:
                print(f"[gpu] monitor unavailable: {exc}", flush=True)
                return
            self._gpu_stop.wait(self.settings.gpu_poll_seconds)


def settings_for_wandb(settings: Settings) -> dict[str, Any]:
    """Return only non-secret configuration values for experiment tracking."""
    data = settings.model_dump()
    return {key: str(value) if isinstance(value, Path) else value for key, value in data.items()}
