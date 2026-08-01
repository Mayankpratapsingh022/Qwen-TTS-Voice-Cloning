# Qwen TTS voice cloning

Fine-tune Qwen3-TTS on an authorized voice, generate speech from the resulting checkpoint, and compare it with Qwen's zero-shot voice-cloning baseline. This is a standalone repository. It does not use, modify, or depend on `cosyvoice3-finetune`.

The workflow is designed for a private dataset stored on Hugging Face and a GPU workspace with persistent storage. It keeps data, generated audio, checkpoints, and credentials out of Git.

## Why fine-tune a voice model

A hosted voice service is often the quickest way to produce speech. Fine-tuning an open Qwen model makes sense when control matters more than instant setup:

- Your recordings and generated outputs can remain in your own storage and GPU environment.
- You control the training data, transcript corrections, checkpoint retention, and model version.
- The workflow is reproducible. You can compare datasets and training settings rather than relying on a changing hosted voice.
- It is useful for research, internal tools, offline or self-hosted products, and teams that need to understand the model they are deploying.
- You are not tied to a per-character API or a vendor-specific inference endpoint.

There is a trade-off. Fine-tuning needs clean recordings, accurate transcripts, GPU time, and listening-based evaluation. A training loss that goes down does not prove that a voice clone sounds more like the speaker.

## What is included

- Qwen3-TTS 12Hz 1.7B Base fine-tuning with Qwen's official training script.
- Private Hugging Face dataset import that preserves Kaldi `train`, `cv`, and `holdout` splits.
- A smoke test before the main training run.
- Live terminal output for downloads, audio tokenization, training, GPU health, and evaluation.
- Weights & Biases tracking for loss and GPU metrics when `WANDB_API_KEY` is available.
- Zero-shot and fine-tuned generation on an untouched holdout set.
- WER, speaker-similarity, speaking-rate, and listener-vote comparison.
- A private local folder for listening samples from each experiment.

## Requirements

- Python 3.10 or later.
- A CUDA-capable NVIDIA GPU for training and practical inference.
- A Hugging Face token if your dataset is private.
- A Weights & Biases API key if you want online monitoring.
- A persistent volume when using a cloud GPU. All project files should live under `/workspace` on RunPod.

An RTX 5090 with 32 GB VRAM was sufficient for the included 1.7B experiment with batch size two. Larger GPUs reduce pressure or wall-clock time but do not improve the voice merely because they have more VRAM.

## Quick start

Clone the repository and create an isolated environment:

```bash
git clone https://github.com/Mayankpratapsingh022/Qwen-TTS-Voice-Cloning.git
cd Qwen-TTS-Voice-Cloning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[train,eval,dev]"
cp .env.example .env
```

Put credentials in `.env`, never in source code or Git:

```bash
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key_here
```

On RunPod, set these as Secrets or template environment variables instead. Verify that the keys are visible without printing them:

```bash
test -n "$HF_TOKEN" && echo "HF token configured" || echo "HF token missing"
test -n "$WANDB_API_KEY" && echo "W&B key configured" || echo "W&B key missing"
```

## Prepare a custom dataset

Use one speaker only. Each clip should be one to 30 seconds, clear, and paired with the exact words spoken. Keep background music, room echo, clipping, other speakers, and aggressive platform compression out of the training data whenever possible.

For local recordings, create a CSV with `audio,text` columns and run:

```bash
qwen-voiceclone data prepare \
  --raw-dir data/raw \
  --metadata metadata.csv \
  --reference-audio data/reference.wav \
  --dest data/processed/my_voice
```

For a Kaldi-style Hugging Face dataset, the expected source layout is `train`, `cv`, and `holdout` directories containing `text`, `wav.scp`, and speaker metadata. Import it before training:

```bash
qwen-voiceclone data import-hf \
  --repo-id YOUR_HF_ACCOUNT/YOUR_VOICE_DATASET \
  --download-dir data/huggingface/my_voice \
  --dest data/processed/my_voice
```

The importer creates `train.jsonl`, `validation.jsonl`, `holdout.jsonl`, `reference.txt`, `reference_utterance.txt`, and `skipped.tsv`. The private source dataset and all processed manifests are ignored by Git.

## Run on a cloud GPU

Start a GPU Pod with persistent `/workspace` storage. Then clone this repository into `/workspace` and run:

```bash
cd /workspace/Qwen-TTS-Voice-Cloning
bash scripts/runpod_setup.sh
```

The setup script reports the GPU, CUDA and Torch versions, Qwen clone progress, package installation, FlashAttention installation, and model-weight download progress. It stores caches under `/workspace/.cache`, so they survive a Pod restart.

Load the RTX 5090 budget profile before training when using that GPU:

```bash
set -a
source profiles/runpod-rtx-5090-32gb.env
set +a
qwen-voiceclone budget show
```

Import your private dataset if it is not already present on the persistent volume, then start the one-epoch experiment:

```bash
qwen-voiceclone data import-hf \
  --repo-id YOUR_HF_ACCOUNT/YOUR_VOICE_DATASET \
  --download-dir data/huggingface/my_voice \
  --dest data/processed/my_voice

bash scripts/runpod_train_and_eval.sh
```

The driver runs a smoke test, the main fine-tuning run, zero-shot holdout generation, checkpoint holdout generation, and scoring. Its output remains visible in the terminal and is also saved under `runs/<run-id>/`.

## Generate a voice sample

After training, generate any new sentence from the checkpoint:

```bash
qwen-voiceclone generate \
  --checkpoint runs/your-run/finetune/checkpoint-epoch-0 \
  --speaker my_voice \
  --text "The windmill dances gently to the tune of the breeze." \
  --output artifacts/manual-tests/windmill.wav
```

The output file is a WAV. `artifacts/` is ignored by Git, so generated speech does not become public by accident.

## Evaluate a fine-tuned model

The driver evaluates audio generated from the holdout split. It compares the fine-tuned checkpoint with Qwen Base zero-shot voice cloning using:

| Check | Why it matters |
| --- | --- |
| Word error rate | Checks whether generated speech says the requested text. Lower is better. |
| Speaker similarity | Estimates whether generated speech resembles the reference speaker. Higher is better. |
| Words per second | Flags a model that speaks at an implausibly different pace. |
| Blind listener votes | Captures voice quality and accent changes that automated metrics miss. |

Create listener votes with one `winner` column containing either `baseline` or `candidate`, then compare:

```bash
qwen-voiceclone eval compare \
  --candidate runs/your-run/finetune_score.json \
  --baseline runs/your-run/baseline_score.json \
  --listener-votes runs/listener_votes.csv
```

Do not add epochs just because the loss decreases. Continue only when the holdout results and blind listening test show a real improvement.

## Completed experiment

This repository contains the record of one private-dataset experiment using `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.

| Item | Result |
| --- | --- |
| Training set | 240 clips, 61.60 minutes |
| Validation set | 14 clips, 3.39 minutes |
| Untouched holdout | 20 clips, 5.80 minutes |
| Training | 1 epoch, batch size 2, learning rate `2e-6` |
| GPU | NVIDIA GeForce RTX 5090, 32 GB VRAM |
| Peak observed VRAM | 25.3 GB |
| Loss | 14.50 at step 0, 10.05 at step 110 |

| Metric | Zero-shot baseline | Fine-tuned checkpoint | Better result |
| --- | ---: | ---: | --- |
| Word error rate | 10.07% | 7.64% | Fine-tuned |
| Speaker similarity | 0.857 | 0.810 | Baseline |
| Words per second | 3.42 | 3.03 | Both usable |

The fine-tuned checkpoint pronounced the holdout text more accurately, but it scored lower on speaker similarity and lost two of three listener votes. It was kept as an experiment rather than accepted as the final model. The reported accent shift is a reason to audit transcripts and recording quality before another run.

## Published listening samples

`InferenceOutput/` contains five shareable samples from the completed experiment:

- the original windmill recording
- the supplied ElevenLabs output for the same sentence
- the Qwen fine-tuned windmill output
- two additional Qwen fine-tuned inference samples

See [InferenceOutput/README.md](InferenceOutput/README.md) for filenames and context. Each sample now has an MP4 version with a white background, animated waveform, and a single source label. GitHub renders those videos directly in the repository interface. Only these five audio files and their five matching videos are versioned. The training dataset, checkpoints, holdout outputs, and all other generated audio remain ignored by Git.

| Sample | Audio | GitHub-friendly video |
| --- | --- | --- |
| Original voice reference | [Audio](InferenceOutput/01_original_voice_sample.mp3) | [MP4](InferenceOutput/videos/01_original_voice_reference.mp4) |
| ElevenLabs voice clone | [Audio](InferenceOutput/02_elevenlabs_voice_clone_sample.mp3) | [MP4](InferenceOutput/videos/02_elevenlabs_voice_clone.mp4) |
| Qwen fine-tuned voice clone | [Audio](InferenceOutput/03_qwen_finetuned_model_inference_windmill.wav) | [MP4](InferenceOutput/videos/03_qwen_finetuned_voice_clone.mp4) |
| Qwen fine-tuned inference: ML Algorithm Visualizer | [Audio](InferenceOutput/04_qwen_finetuned_model_inference_ml_algorithm_visualizer.wav) | [MP4](InferenceOutput/videos/04_qwen_finetuned_inference_ml_algorithm_visualizer.mp4) |
| Qwen fine-tuned inference: Word Embeddings | [Audio](InferenceOutput/05_qwen_finetuned_model_inference_word_embeddings.wav) | [MP4](InferenceOutput/videos/05_qwen_finetuned_inference_word_embeddings.mp4) |

## Files created by a run

Each `runs/<run-id>/` directory contains the checkpoint, generated holdout WAVs, score reports, command history, GPU CSV samples, and complete logs. The model weights, W&B files, dataset, audio, and checkpoints are ignored by Git.

Before stopping a cloud Pod, copy any checkpoint or audio you want to keep locally. Stopping the Pod is safe when those files live under persistent `/workspace`; terminating its volume is not.

## Next experiment

Start from the original Qwen Base model rather than continuing the current one-epoch checkpoint. First audit transcript alignment on representative clips, especially names, numbers, abbreviations, technical terms, and mixed-language speech. Then add clean, consistently recorded English samples if needed and repeat the same one-epoch holdout comparison.

The text-projection modification discussed in some community projects is not an official validated fix for the 1.7B model. Treat it as a separate experiment after establishing a stronger data-quality baseline.
