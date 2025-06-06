import wave
from pathlib import Path

from wyoming.audio import AudioChunk

from easy_audio_interfaces.types.common import PathLike


def audio_chunk_from_file(file_path: PathLike) -> AudioChunk:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with wave.open(str(file_path), "rb") as wav_file:
        audio_data = wav_file.readframes(wav_file.getnframes())
        return AudioChunk(
            audio=audio_data,
            rate=wav_file.getframerate(),
            width=wav_file.getsampwidth(),
            channels=wav_file.getnchannels(),
        )
