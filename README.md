## Easywhisper

Easywhisper is a transcription library which decomposes the inference pipeline into independent and parallelizable components (VAD, transcription, feature/emission extraction, forced alignment). The library is therefore well suited for transcribing large archives of audio files efficiently. Supports both `ctranslate2` (faster-whisper) and `Hugging Face` backends for inference. The library features:

* Batch inference support for both wav2vec2 and Whisper models.
* Parallel loading of audio files for efficient batch processing.
* GPU accelerated forced alignment.
* Saves wav2vec2 emmissions to disk for flexible parallel processing on CPU/GPU.

