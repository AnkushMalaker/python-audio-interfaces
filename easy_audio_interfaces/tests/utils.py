import asyncio
import math
import struct
from typing import AsyncGenerator

from wyoming.audio import AudioChunk


def create_sine_wave_audio_chunk(
    duration_ms: int,
    frequency: int = 440,
    sample_rate: int = 44100,
    channels: int = 1,
    width: int = 2,
) -> AudioChunk:
    """Create a sine wave as an AudioChunk."""
    num_samples = int(sample_rate * duration_ms / 1000)
    audio_data = bytearray()

    for i in range(num_samples):
        # Generate sine wave sample
        t = i / sample_rate
        sample = int(32767 * math.sin(2 * math.pi * frequency * t))

        # Pack as 16-bit signed integer (width=2)
        sample_bytes = struct.pack("<h", sample)

        # For stereo, duplicate the sample
        for _ in range(channels):
            audio_data.extend(sample_bytes)

    return AudioChunk(audio=bytes(audio_data), rate=sample_rate, width=width, channels=channels)


async def async_generator(
    audio_chunk: AudioChunk, chunk_duration_ms: int = 1000
) -> AsyncGenerator[AudioChunk, None]:
    """Split an AudioChunk into smaller chunks for async streaming."""
    # Calculate bytes per millisecond
    bytes_per_ms = (audio_chunk.rate * audio_chunk.width * audio_chunk.channels) // 1000
    chunk_size_bytes = chunk_duration_ms * bytes_per_ms

    audio_data = audio_chunk.audio

    for i in range(0, len(audio_data), chunk_size_bytes):
        chunk_data = audio_data[i : i + chunk_size_bytes]
        if chunk_data:  # Only yield non-empty chunks
            chunk = AudioChunk(
                audio=chunk_data,
                rate=audio_chunk.rate,
                width=audio_chunk.width,
                channels=audio_chunk.channels,
            )
            await asyncio.sleep(0.0)  # simulate async
            yield chunk
