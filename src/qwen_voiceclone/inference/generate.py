"""Generate audio from a selected fine-tuned checkpoint."""

from __future__ import annotations

from pathlib import Path

from qwen_voiceclone.data.manifest import read_jsonl


def generate_custom_voice(checkpoint: Path, text: str, speaker: str, output: Path) -> None:
    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError("run generation in the GPU environment after installing qwen-tts") from exc
    model = Qwen3TTSModel.from_pretrained(
        str(checkpoint), device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    wavs, sample_rate = model.generate_custom_voice(text=text, language="English", speaker=speaker)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wavs[0], sample_rate)


def generate_zero_shot(model_id: str, text: str, reference_audio: Path, reference_text: str, output: Path) -> None:
    """Generate the required reference-conditioned baseline from the released Base model."""
    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError("run generation in the GPU environment after installing qwen-tts") from exc
    model = Qwen3TTSModel.from_pretrained(
        model_id, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    wavs, sample_rate = model.generate_voice_clone(text=text, language="English", ref_audio=str(reference_audio), ref_text=reference_text)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wavs[0], sample_rate)


def generate_holdout(
    model_path: str,
    manifest: Path,
    reference_text: str | None,
    output_dir: Path,
    mode: str,
    language: str,
    speaker: str | None = None,
) -> None:
    """Produce visible, deterministic holdout outputs while loading the model only once."""
    try:
        import soundfile as sf
        import torch
        from qwen_tts import Qwen3TTSModel
    except ImportError as exc:
        raise RuntimeError("run generation in the GPU environment after installing qwen-tts") from exc
    if mode not in {"baseline", "checkpoint"}:
        raise ValueError("mode must be baseline or checkpoint")
    if mode == "baseline" and not reference_text:
        raise ValueError("reference_text is required for baseline generation")
    print(f"[eval] loading {mode} model: {model_path}", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        model_path, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    )
    rows = read_jsonl(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        print(f"[eval] generating {index}/{len(rows)}: {row['id']}", flush=True)
        if mode == "baseline":
            wavs, sample_rate = model.generate_voice_clone(
                text=row["text"], language=language, ref_audio=row["ref_audio"], ref_text=reference_text
            )
        else:
            wavs, sample_rate = model.generate_custom_voice(text=row["text"], language=language, speaker=speaker)
        sf.write(output_dir / f"{row['id']}.wav", wavs[0], sample_rate)
