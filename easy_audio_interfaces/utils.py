import wave
from pathlib import Path
from typing import BinaryIO

from wyoming.audio import AudioChunk

from easy_audio_interfaces.types.common import PathLike


def audio_chunk_from_source(source: PathLike | BinaryIO) -> AudioChunk:
    """Load entire audio from file or BytesIO into a single AudioChunk. Use WaveFileStreamer for streaming."""
    if isinstance(source, (str, Path)):
        file_path = Path(source)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        source = str(file_path)

    with wave.open(source, "rb") as wav_file:  # type: ignore
        audio_data = wav_file.readframes(wav_file.getnframes())
        return AudioChunk(
            audio=audio_data,
            rate=wav_file.getframerate(),
            width=wav_file.getsampwidth(),
            channels=wav_file.getnchannels(),
        )
