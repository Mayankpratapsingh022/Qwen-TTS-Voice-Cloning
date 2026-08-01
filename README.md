# Qwen TTS Voice Cloning

An open-source workflow for fine-tuning Qwen3-TTS on one authorized voice. It prepares private audio data, runs a controlled GPU training job, records live metrics, and compares the result with Qwen's zero-shot voice-cloning baseline.

This project is separate from `cosyvoice3-finetune`. It does not read from, write to, or depend on that folder.

## What this project does

- Imports a private Hugging Face dataset or prepares local recordings.
- Uses Qwen3-TTS 12Hz 1.7B Base for single-speaker fine-tuning.
- Runs on a RunPod GPU with visible download and training output.
- Records loss and GPU health in Weights & Biases when configured.
- Keeps terminal logs, GPU CSV data, commands, checkpoints, and generated audio on the persistent volume.
- Evaluates zero-shot and fine-tuned audio on an untouched holdout split.

## Voice and data safety

Only train on a voice that you own or have clear permission to clone. A fine-tuned model can closely match a speaker, but no model can promise an identical result for every sentence, style, or recording condition. The holdout comparison is part of the workflow so the fine-tuned model is not accepted only because its training loss dropped.

## Local setup

```bash
cd qwen3tts-voiceclone
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[train,eval,dev]"
cp .env.example .env
```

Keep `HF_TOKEN` in `.env` only. Do not commit it or paste it into commands.

## Prepare a local dataset

For a fresh set of recordings, create a CSV with `audio,text` columns. Put each recording session in its own directory under `data/raw`. The preparation step keeps complete sessions apart when it creates train, validation, and holdout splits.

```bash
qwen-voiceclone data prepare \
  --raw-dir data/raw \
  --metadata metadata.csv \
  --reference-audio data/reference.wav \
  --dest data/processed/my_voice
```

Audio should be mono, at least 16 kHz, one to 30 seconds per clip, and contain one speaker. Clean transcripts matter as much as clean audio.

## Import the Mayank dataset

`Mayank022/voiceclone-mayank-dataset` already has Kaldi-style `train`, `cv`, and `holdout` splits. The importer downloads the WAV chunks into this project, rewrites the old absolute paths, validates the audio, and keeps those split boundaries intact.

```bash
qwen-voiceclone data import-hf \
  --repo-id Mayank022/voiceclone-mayank-dataset \
  --download-dir data/huggingface/mayank \
  --dest data/processed/mayank
```

The processed directory contains:

- `train.jsonl`
- `validation.jsonl`
- `holdout.jsonl`
- `reference.txt`, the exact transcript for the selected reference clip
- `reference_utterance.txt`
- `skipped.tsv`, which records any rejected clips

The imported dataset currently contains 240 train clips, 14 validation clips, and 20 holdout clips. No clips were rejected during import.

## RunPod training

Use a RunPod PyTorch template with one NVIDIA GPU and a persistent `/workspace` volume. Reserve at least 50 GB. All caches, downloads, logs, checkpoints, and audio outputs stay under `/workspace`, so they survive a Pod restart.

Before creating the Pod, add these RunPod Secrets:

- `WANDB_API_KEY` for the private W&B dashboard
- `HF_TOKEN` only if you plan to import the private dataset on the Pod

Reference the secrets through the Pod template environment variables. Do not put either key in the repository. If W&B is unavailable during a run, training continues and retains its local logs and checkpoints.

Copy this project, without `.venv`, to `/workspace/qwen3tts-voiceclone`. If you do not copy `data/huggingface/mayank`, import the dataset on the Pod after setup.

```bash
cd /workspace/qwen3tts-voiceclone
bash scripts/runpod_setup.sh
```

The setup script prints and saves every long operation. You will see the official Qwen repository clone, package installation, FlashAttention installation, Qwen model download, GPU details, CUDA version, Torch version, and the Qwen Git commit. Its complete output is saved in `runs/setup-*/setup.log`.

If needed, import the private dataset on the Pod:

```bash
qwen-voiceclone data import-hf \
  --repo-id Mayank022/voiceclone-mayank-dataset \
  --download-dir data/huggingface/mayank \
  --dest data/processed/mayank
```

Start the controlled training and evaluation run:

```bash
bash scripts/runpod_train_and_eval.sh
```

## RTX PRO 6000 Blackwell 96GB

The RTX PRO 6000 Blackwell 96GB is a good choice for this project. The extra VRAM is not required for a 1.7B model with batch size two, so it does not change voice quality. It does give the run more memory headroom and can reduce wall-clock time.

At $1.89 per hour, use the included profile. It limits smoke testing, fine-tuning, and evaluation to 5.2 GPU hours, or $9.83 maximum. Set the RunPod auto-stop timer to 5 hours and 12 minutes.

```bash
cd /workspace/qwen3tts-voiceclone
set -a
source profiles/runpod-rtx-pro-6000-96gb.env
set +a
bash scripts/runpod_setup.sh
bash scripts/runpod_train_and_eval.sh
```

The profile is in `profiles/runpod-rtx-pro-6000-96gb.env`. The setup script confirms that the Pod can see the GPU and that CUDA, Torch, and FlashAttention load before training starts.

## Monitoring and files to inspect

The RunPod driver writes all output to the web terminal and to the run directory. Watch the terminal for download progress, Qwen loss lines, GPU memory use, temperature, and evaluation progress.

Each run contains:

- `driver.log`, the full driver output
- `logs/tokenize.log` and `logs/sft.log`, output from Qwen's data and training scripts
- `logs/gpu.csv`, sampled GPU utilization, VRAM, temperature, and power
- `logs/events.jsonl` and `logs/commands.log`
- `metadata.json`, model, budget, and training settings
- `checkpoint-epoch-0`, the first fine-tuned model checkpoint
- `baseline_audio` and `finetune_audio`, generated holdout WAVs
- `baseline_score.json` and `finetune_score.json`

When `WANDB_API_KEY` is available, the run prints its W&B URL. The dashboard records training loss and GPU metrics. To retain W&B files locally without sending them online, set `QVC_WANDB_MODE=offline`.

## Evaluation

The RunPod driver performs these steps after training:

1. Generates every holdout sentence with Qwen Base using the selected reference audio.
2. Generates the same sentences with `checkpoint-epoch-0`.
3. Scores both sets with ASR word error rate, SpeechBrain speaker similarity, and speaking rate.
4. Leaves the audio and JSON reports in the run directory for listening tests.

Do not choose a checkpoint from metrics alone. Run a blind listening test and record listener choices in a CSV with one `winner` column containing `candidate` or `baseline`.

```bash
qwen-voiceclone eval compare \
  --candidate runs/your-run/finetune_score.json \
  --baseline runs/your-run/baseline_score.json \
  --listener-votes runs/listener_votes.csv
```

The comparison accepts a candidate only when it meets the objective checks and wins at least two-thirds of listener votes.

## Manual commands

These commands are useful when you want to run one step at a time:

```bash
qwen-voiceclone budget show
qwen-voiceclone models prefetch
qwen-voiceclone train smoke --manifest data/processed/mayank/train.jsonl --output-dir runs/smoke
qwen-voiceclone train run --manifest data/processed/mayank/train.jsonl --output-dir runs/first-epoch --epochs 1 --learning-rate 2e-6
qwen-voiceclone eval generate-baseline --manifest data/processed/mayank/holdout.jsonl --reference-text-file data/processed/mayank/reference.txt --output-dir artifacts/baseline
qwen-voiceclone eval generate-checkpoint --checkpoint runs/first-epoch/checkpoint-epoch-0 --manifest data/processed/mayank/holdout.jsonl --output-dir artifacts/finetune
```

Start with one epoch. Add more epochs only if the first checkpoint clearly beats the zero-shot baseline on the holdout audio and listening test.
