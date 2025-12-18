from pathlib import Path

import torch
from easyalign.data.dataset import AudioFileDataset
from easyalign.utils import save_metadata_json
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from easywhisper.data.collators import transcribe_collate_fn


def transcribe(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    file_dataloader: torch.utils.data.DataLoader,
    output_dir: str = "output/transcriptions",
):
    for features in file_dataloader:
        slice_dataset = features[0]["dataset"]
        metadata = features[0]["dataset"].metadata
        transcription_texts = []

        feature_dataloader = torch.utils.data.DataLoader(
            slice_dataset,
            batch_size=4,
            num_workers=2,
            prefetch_factor=2,
            collate_fn=transcribe_collate_fn,
        )

        for batch in feature_dataloader:
            with torch.inference_mode():
                batch = batch["features"].to("cuda").half()
                predicted_ids = model.generate(
                    batch,
                    return_dict_in_generate=True,
                    task="transcribe",
                    language="sv",
                    output_scores=True,
                    max_length=250,
                )

                transcription = processor.batch_decode(
                    predicted_ids["sequences"], skip_special_tokens=True
                )

                transcription_texts.extend(transcription)

        for i, speech in enumerate(metadata.speeches):
            for j, chunk in enumerate(speech.chunks):
                chunk.text = transcription_texts[j]

        # Write final transcription to file with msgspec serialization
        output_path = Path(output_dir) / Path(metadata.audio_path).with_suffix(".json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_metadata_json(metadata, output_dir=output_dir)
