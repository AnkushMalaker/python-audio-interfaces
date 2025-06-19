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

        # Flag to track if we've initialized from first chunk
        self._initialized = False

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

    def _initialize_from_first_chunk(self, chunk: AudioChunk):
        """Initialize unspecified parameters from the first chunk."""
        if not self._initialized:
            if self._resample_channels is None:
                self._resample_channels = chunk.channels
                logger.info(f"Inferred resample_channels from input: {self._resample_channels}")

            if self._resample_width is None:
                self._resample_width = chunk.width
                logger.info(f"Inferred resample_width from input: {self._resample_width}")

            self._initialized = True

    async def process(self, input_stream: AudioStream) -> AudioStream:
        async for chunk in input_stream:
            # Initialize unspecified parameters from first chunk
            self._initialize_from_first_chunk(chunk)

            # Now we know all parameters are set
            assert self._resample_channels is not None
            assert self._resample_width is not None

            # Check if any conversion is needed
            if (
                chunk.rate == self._resample_rate
                and chunk.channels == self._resample_channels
                and chunk.width == self._resample_width
            ):
                # No conversion needed, pass through
                yield chunk
                continue

            # Convert to numpy array
            audio_array = self._audio_chunk_to_numpy(chunk)

            # Handle channel conversion first
            if chunk.channels != self._resample_channels:
                if chunk.channels == 2 and self._resample_channels == 1:
                    # Convert stereo to mono by averaging channels
                    if audio_array.ndim == 2:
                        # Average and keep as integer (with proper scaling to prevent overflow)
                        audio_array = np.mean(audio_array, axis=1, dtype=audio_array.dtype)
                elif chunk.channels == 1 and self._resample_channels == 2:
                    # Convert mono to stereo by duplicating channel
                    audio_array = np.column_stack([audio_array, audio_array])
                else:
                    logger.warning(
                        f"Unsupported channel conversion: {chunk.channels} -> {self._resample_channels}"
                    )

            # Resample if needed (handles both mono and multi-channel audio)
            if chunk.rate != self._resample_rate:
                # Convert to supported format for soxr if needed
                if audio_array.dtype == np.int8:
                    # Convert int8 to int16 for resampling
                    resample_input = audio_array.astype(np.int16) << 8
                    resampled = soxr.resample(
                        resample_input, chunk.rate, self._resample_rate, quality=self._quality
                    )
                    # Convert back to int8 if target is int8
                    if self._resample_width == 1:
                        resampled = (resampled >> 8).astype(np.int8)
                else:
                    resampled = soxr.resample(
                        audio_array, chunk.rate, self._resample_rate, quality=self._quality
                    )
            else:
                resampled = audio_array

            # Handle bit depth conversion if needed
            if chunk.width != self._resample_width:
                if chunk.width == 4 and self._resample_width == 2:
                    # Convert 32-bit to 16-bit by right-shifting
                    resampled_int = (resampled >> 16).astype(np.int16)
                elif chunk.width == 2 and self._resample_width == 4:
                    # Convert 16-bit to 32-bit by left-shifting
                    resampled_int = resampled.astype(np.int32) << 16
                elif chunk.width == 4 and self._resample_width == 1:
                    # Convert 32-bit to 8-bit
                    resampled_int = (resampled >> 24).astype(np.int8)
                elif chunk.width == 2 and self._resample_width == 1:
                    # Convert 16-bit to 8-bit
                    resampled_int = (resampled >> 8).astype(np.int8)
                elif chunk.width == 1 and self._resample_width == 2:
                    # Convert 8-bit to 16-bit
                    resampled_int = resampled.astype(np.int16) << 8
                elif chunk.width == 1 and self._resample_width == 4:
                    # Convert 8-bit to 32-bit
                    resampled_int = resampled.astype(np.int32) << 24
                else:
                    raise ValueError(
                        f"Unsupported bit depth conversion: {chunk.width} -> {self._resample_width}"
                    )
            else:
                resampled_int = resampled

            # Create new AudioChunk with resampled data
            resampled_chunk = self._numpy_to_audio_chunk(
                resampled_int,
                self._resample_rate,
                self._resample_width,
                self._resample_channels,
            )

            yield resampled_chunk

    async def open(self):
        self._initialized = False

    async def close(self):
        self._initialized = False


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
