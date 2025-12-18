from pathlib import Path

import ctranslate2
import soundfile as sf
import torch
from easyalign.data.collators import audiofile_collate_fn, transcribe_collate_fn
from easyalign.data.dataset import AudioFileDataset, JSONMetadataDataset
from transformers import WhisperProcessor

audio, sample_rate = sf.read("data/YS_sr_p1_2003-09-02_0525_0600.wav")
# Compute the features of the first 30 seconds of audio.
processor = WhisperProcessor.from_pretrained("KBLab/kb-whisper-large")
# inputs = processor(audio, return_tensors="np", sampling_rate=16000)
# features = ctranslate2.StorageView.from_array(inputs.input_features)

model = ctranslate2.models.Whisper("models/kb-whisper-large", device="cuda")

json_dataset = JSONMetadataDataset(json_paths=list(Path("output/vad").rglob("*.json")))

file_dataset = AudioFileDataset(
    metadata=json_dataset,
    processor=processor,
    sample_rate=16000,
    chunk_size=30,
    alignment_strategy="chunk",
)

file_dataloader = torch.utils.data.DataLoader(
    file_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=audiofile_collate_fn,
    num_workers=2,
    prefetch_factor=2,
)

# dataset = next(iter(file_dataloader))

prompt = processor.tokenizer.convert_tokens_to_ids(
    [
        "<|startoftranscript|>",
        "<|transcribe|>",
        "<|notimestamps|>",  # Remove this token to generate timestamps.
    ]
)


for dataset in file_dataloader:
    slice_dataset = dataset[0]["dataset"]

    feature_dataloader = torch.utils.data.DataLoader(
        slice_dataset,
        batch_size=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=transcribe_collate_fn,
        num_workers=2,
        prefetch_factor=2,
    )
    for batch in feature_dataloader:
        features = batch["features"].numpy()
        batch_size = features.shape[0]
        features = ctranslate2.StorageView.from_array(features)
        outputs = model.generate(features, [prompt] * batch_size, beam_size=5)
        sequences = [result.sequences_ids[0] for result in outputs]
        transcription = processor.batch_decode(sequences, skip_special_tokens=True)
        print("Transcription: %s" % transcription)

# Detect the language.
results = model.detect_language(features)
language, probability = results[0][0]
print("Detected language %s with probability %f" % (language, probability))


# Describe the task in the prompt.
# See the prompt format in https://github.com/openai/whisper.
prompt = processor.tokenizer.convert_tokens_to_ids(
    [
        "<|startoftranscript|>",
        language,
        "<|transcribe|>",
        "<|notimestamps|>",  # Remove this token to generate timestamps.
    ]
)

results = model.generate(
    features,
    [prompt],
    beam_size=5,
    patience=1.0,
    num_hypotheses=1,
    length_penalty=1.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    max_length=448,
    return_scores=False,
    return_logits_vocab=False,
    return_no_speech_prob=False,
    max_initial_timestamp_index=0,
    suppress_blank=True,
    suppress_tokens=[-1],
    sampling_topk=1,
    sampling_temperature=1,
)
transcription = processor.decode(results[0].sequences_ids[0])

print("Transcription: %s" % transcription)
