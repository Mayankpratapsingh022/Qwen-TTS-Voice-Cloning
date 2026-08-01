# Published listening samples

This folder contains five public listening samples from the first Qwen3-TTS fine-tuning experiment. The owner authorized publication of the original recording and generated voice-clone outputs below. All other training audio, evaluation outputs, and checkpoints remain ignored by Git.

The windmill sentence is: "The windmill dances gently to the tune of the breeze."

| File | Source | Purpose |
| --- | --- | --- |
| [Original voice sample](01_original_voice_sample.mp3) | Original recording | Ground-truth reference for the windmill sentence, converted from the supplied M4A. |
| [ElevenLabs voice-clone sample](02_elevenlabs_voice_clone_sample.mp3) | ElevenLabs output supplied by the owner | External comparison for the same sentence. |
| [Qwen fine-tuned windmill inference](03_qwen_finetuned_model_inference_windmill.wav) | Qwen 1-epoch checkpoint | Manual inference from `checkpoint-epoch-0`. |
| [Qwen fine-tuned ML Algorithm Visualizer inference](04_qwen_finetuned_model_inference_ml_algorithm_visualizer.wav) | Qwen 1-epoch checkpoint | Fine-tuned inference sample. |
| [Qwen fine-tuned Word Embeddings inference](05_qwen_finetuned_model_inference_word_embeddings.wav) | Qwen 1-epoch checkpoint | Fine-tuned inference sample. |

The automated baseline-versus-fine-tuned holdout comparison remains documented in the repository README. These five files are a public listening sample, not a replacement for a blind evaluation.
