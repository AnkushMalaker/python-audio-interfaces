import logging

import numpy as np
import soxr
from wyoming.audio import AudioChunk

from easy_audio_interfaces.base_interfaces import (
    AudioSink,
    AudioSource,
    ProcessingBlock,
)
from easy_audio_interfaces.types.common import AudioStream

logger = logging.getLogger(__name__)


# TODO
class StreamFromCommand(AudioSource):
    def __init__(self, command: str):
        self._command = command

    async def open(self):
        ...

    async def close(self):
        ...


class ResamplingBlock(ProcessingBlock):
    def __init__(
        self,
        resample_rate: int,
        resample_channels: int | None = None,
        resample_width: int | None = None,
        quality: str = "HQ",  # HQ, VHQ, MQ, LQ, or QQ
    ):
        self._resample_rate = resample_rate
        self._resample_channels = resample_channels
        self._resample_width = resample_width
        self._quality = quality

    @property
    def sample_rate(self) -> int:
        return self._resample_rate

    def _audio_chunk_to_numpy(self, chunk: AudioChunk) -> np.ndarray:
        """Convert AudioChunk bytes to numpy array."""
        dtype: type[np.signedinteger] | type[np.floating]
        # Determine the numpy dtype based on audio width
        if chunk.width == 1:
            dtype = np.int8
        elif chunk.width == 2:
            dtype = np.int16
        elif chunk.width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported audio width: {chunk.width}")

        # Convert bytes to numpy array
        audio_array = np.frombuffer(chunk.audio, dtype=dtype)

        # Reshape for multiple channels if needed
        if chunk.channels > 1:
            audio_array = audio_array.reshape(-1, chunk.channels)

        return audio_array

    def _numpy_to_audio_chunk(
        self, audio_array: np.ndarray, rate: int, width: int, channels: int
    ) -> AudioChunk:
        """Convert numpy array back to AudioChunk."""
        # Determine the numpy dtype based on audio width
        dtype: type[np.signedinteger] | type[np.floating]
        if width == 1:
            dtype = np.int8
        elif width == 2:
            dtype = np.int16
        elif width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported audio width: {width}")

        # Ensure the array is the correct dtype
        audio_array = audio_array.astype(dtype)

        # Convert back to bytes
        audio_bytes = audio_array.tobytes()

        return AudioChunk(
            audio=audio_bytes,
            rate=rate,
            width=width,
            channels=channels,
        )

    async def process(self, input_stream: AudioStream) -> AudioStream:

        async for chunk in input_stream:
            # Check if any conversion is needed
            if self._resample_channels is None:
                self._resample_channels = chunk.channels
            if self._resample_width is None:
                self._resample_width = chunk.width

            if (
                chunk.rate == self._resample_rate
                and chunk.channels == self._resample_channels
                and chunk.width == self._resample_width
            ):
                # No conversion needed
                yield chunk
                continue

            # Convert to numpy array
            audio_array = self._audio_chunk_to_numpy(chunk)

            # Convert to float for processing (normalize to [-1.0, 1.0])
            if chunk.width == 1:
                audio_float = audio_array.astype(np.float32) / np.iinfo(np.int8).max
            elif chunk.width == 2:
                audio_float = audio_array.astype(np.float32) / np.iinfo(np.int16).max
            elif chunk.width == 4:
                audio_float = audio_array.astype(np.float32) / np.iinfo(np.int32).max
            else:
                raise ValueError(f"Unsupported input audio width: {chunk.width}")

            # Handle channel conversion
            if chunk.channels != self._resample_channels:
                if chunk.channels == 2 and self._resample_channels == 1:
                    # Convert stereo to mono by averaging channels
                    if audio_float.ndim == 2:
                        audio_float = np.mean(audio_float, axis=1)
                elif chunk.channels == 1 and self._resample_channels == 2:
                    # Convert mono to stereo by duplicating channel
                    audio_float = np.column_stack([audio_float, audio_float])
                else:
                    logger.warning(
                        f"Unsupported channel conversion: {chunk.channels} -> {self._resample_channels}"
                    )

            # Resample if needed (handles both mono and multi-channel audio)
            if chunk.rate != self._resample_rate:
                resampled = soxr.resample(
                    audio_float, chunk.rate, self._resample_rate, quality=self._quality
                )
            else:
                resampled = audio_float

            # Convert back to target format (scale from [-1.0, 1.0] to target bit depth)
            if self._resample_width == 1:
                resampled_int = (resampled * np.iinfo(np.int8).max).astype(np.int8)
            elif self._resample_width == 2:
                resampled_int = (resampled * np.iinfo(np.int16).max).astype(np.int16)
            elif self._resample_width == 4:
                resampled_int = (resampled * np.iinfo(np.int32).max).astype(np.int32)
            else:
                raise ValueError(f"Unsupported target audio width: {self._resample_width}")

            # Create new AudioChunk with resampled data
            resampled_chunk = self._numpy_to_audio_chunk(
                resampled_int, self._resample_rate, self._resample_width, self._resample_channels
            )

            yield resampled_chunk

    async def open(self):
        pass

    async def close(self):
        pass


class RechunkingBlock(ProcessingBlock):
    def __init__(self, *, chunk_size_ms: int | None = None, chunk_size_samples: int | None = None):
        if chunk_size_ms is None and chunk_size_samples is None:
            raise ValueError("Either chunk_size_ms or chunk_size_samples must be provided")
        if chunk_size_ms is not None and chunk_size_samples is not None:
            raise ValueError("Only one of chunk_size_ms or chunk_size_samples can be provided")

        self._chunk_size_ms = chunk_size_ms
        self._chunk_size_samples = chunk_size_samples
        self._buffer = b""
        self._audio_format: AudioChunk | None = None

    def _ms_to_samples(self, ms: int, sample_rate: int) -> int:
        """Convert milliseconds to number of samples."""
        return int((ms * sample_rate) / 1000)

    def _samples_to_bytes(self, samples: int, width: int, channels: int) -> int:
        """Convert number of samples to bytes."""
        return samples * width * channels

    def _get_target_chunk_size_bytes(self, audio_format: AudioChunk) -> int:
        """Get the target chunk size in bytes based on the configured parameters."""
        if self._chunk_size_ms is not None:
            target_samples = self._ms_to_samples(self._chunk_size_ms, audio_format.rate)
        else:
            assert self._chunk_size_samples is not None
            target_samples = self._chunk_size_samples

        return self._samples_to_bytes(target_samples, audio_format.width, audio_format.channels)

    async def _process_audio_stream(self, input_stream: AudioStream) -> AudioStream:
        """Unified processing method that works with both ms and samples by converting to bytes."""
        async for chunk in input_stream:
            # Store audio format from the first chunk
            if self._audio_format is None:
                self._audio_format = chunk

            # Add new audio data to buffer
            self._buffer += chunk.audio

            # Calculate target chunk size in bytes
            target_chunk_size = self._get_target_chunk_size_bytes(chunk)

            # Yield complete chunks
            while len(self._buffer) >= target_chunk_size:
                chunk_audio = self._buffer[:target_chunk_size]
                self._buffer = self._buffer[target_chunk_size:]

                yield AudioChunk(
                    audio=chunk_audio,
                    rate=chunk.rate,
                    width=chunk.width,
                    channels=chunk.channels,
                )

        # Yield any remaining audio in buffer
        if len(self._buffer) > 0 and self._audio_format is not None:
            yield AudioChunk(
                audio=self._buffer,
                rate=self._audio_format.rate,
                width=self._audio_format.width,
                channels=self._audio_format.channels,
            )
            self._buffer = b""

    def process(self, input_stream: AudioStream) -> AudioStream:
        return self._process_audio_stream(input_stream)

    async def open(self):
        self._buffer = b""
        self._audio_format = None

    async def close(self):
        self._buffer = b""
        self._audio_format = None


__all__ = [
    "AudioSource",
    "AudioSink",
    "ResamplingBlock",
    "RechunkingBlock",
    "ProcessingBlock",
]
