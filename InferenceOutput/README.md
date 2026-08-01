# Published listening samples

This folder contains five public listening samples from the first Qwen3-TTS fine-tuning experiment. The owner authorized publication of the original recording and generated voice-clone outputs below. Each has a matching MP4 with a clean white background, animated waveform, and one source label for direct playback on GitHub. All other training audio, evaluation outputs, and checkpoints remain ignored by Git.

The windmill sentence is: "The windmill dances gently to the tune of the breeze."

| Sample | Source | Audio | Video |
| --- | --- | --- | --- |
| Original voice reference | Original recording | [MP3](01_original_voice_sample.mp3) | [MP4](videos/01_original_voice_reference.mp4) |
| ElevenLabs voice clone | ElevenLabs output supplied by the owner | [MP3](02_elevenlabs_voice_clone_sample.mp3) | [MP4](videos/02_elevenlabs_voice_clone.mp4) |
| Qwen fine-tuned voice clone | Qwen 1-epoch checkpoint | [WAV](03_qwen_finetuned_model_inference_windmill.wav) | [MP4](videos/03_qwen_finetuned_voice_clone.mp4) |
| Qwen fine-tuned inference: ML Algorithm Visualizer | Qwen 1-epoch checkpoint | [WAV](04_qwen_finetuned_model_inference_ml_algorithm_visualizer.wav) | [MP4](videos/04_qwen_finetuned_inference_ml_algorithm_visualizer.mp4) |
| Qwen fine-tuned inference: Word Embeddings | Qwen 1-epoch checkpoint | [WAV](05_qwen_finetuned_model_inference_word_embeddings.wav) | [MP4](videos/05_qwen_finetuned_inference_word_embeddings.mp4) |

The automated baseline-versus-fine-tuned holdout comparison remains documented in the repository README. These five files are a public listening sample, not a replacement for a blind evaluation.
