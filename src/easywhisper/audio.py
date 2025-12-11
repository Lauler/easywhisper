import subprocess
from typing import Tuple

import numpy as np


def convert_audio_to_array(input_file: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Convert audio to in-memory numpy array.
    """
    # fmt: off
    command = [
        "ffmpeg",
        "-i", input_file,
        "-f", "s16le",  # raw PCM 16-bit little endian
        "-acodec", "pcm_s16le",
        "-ac", "1",  # mono
        "-ar", str(sample_rate),  # 16 kHz
        "-loglevel", "error",  # suppress output
        "-hide_banner",
        "-nostats",
    ]
    # fmt: on

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {err.decode()}")

    # Convert byte output to numpy array
    audio_array = np.frombuffer(out, dtype=np.int16)

    return audio_array, sample_rate  # (samples, sample_rate)


def convert_audio_to_wav(input_file: str, output_file: str) -> None:
    """
    Convert audio file to WAV format with 16kHz sample rate and mono channel.
    """
    # fmt: off
    command = [
        'ffmpeg',
        '-i', input_file,
        '-ar', '16000',  # Set the audio sample rate to 16kHz
        '-ac', '1',      # Set the number of audio channels to 1 (mono)
        '-c:a', 'pcm_s16le',
        '-loglevel', 'warning',
        '-hide_banner',
        '-nostats',
        '-nostdin',
        output_file
    ]
    # fmt: on
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _, err = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {err.decode()}")
