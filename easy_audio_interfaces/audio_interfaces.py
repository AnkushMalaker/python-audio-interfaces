import asyncio
import logging
from typing import AsyncIterator

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
        quality: str = "VHQ",  # VHQ for better SNR, HQ, MQ, LQ, or QQ
    ):
        """
        Audio resampling block for rate, channel, and bit depth conversion.

        Supports:
        - Sample rate conversion using high-quality SoXR library
        - Channel conversion: mono ↔ stereo
        - Bit depth conversion: 16-bit ↔ 32-bit using arithmetic right-shift
        - Does not support 8-bit audio processing

        32-bit to 16-bit conversion uses arithmetic right-shift (samples32 >> 16)
        which preserves sign and provides deterministic conversion.

        Parameters
        ----------
        resample_rate : int
            Target sample rate in Hz
        resample_channels : int, optional
            Target number of channels (1 for mono, 2 for stereo)
            If None, inferred from first input chunk
        resample_width : int, optional
            Target bit depth in bytes (2 for 16-bit, 4 for 32-bit)
            If None, inferred from first input chunk
        quality : str, default "VHQ"
            SoXR quality setting: "VHQ", "HQ", "MQ", "LQ", or "QQ"
            "VHQ" provides best SNR performance
        """
        # Validate quality parameter
        valid_qualities = {"HQ", "VHQ", "MQ", "LQ", "QQ"}
        if quality not in valid_qualities:
            raise ValueError(f"quality must be one of {valid_qualities}, got: {quality}")

        self._resample_rate = resample_rate
        self._resample_channels = resample_channels
        self._resample_width = resample_width
        self._quality = quality

        # Flag to track if we've initialized from first chunk
        self._initialized = False

        # ResampleStream instance for maintaining filter state between chunks
        self._resample_stream: soxr.ResampleStream | None = None

        # Store chunk properties from first chunk for safe flush
        self._input_width: int | None = None
        self._input_channels: int | None = None

    @property
    def sample_rate(self) -> int:
        return self._resample_rate

    def _int32_to_int16(self, samples32: np.ndarray) -> np.ndarray:
        """
        Convert int32 PCM to int16 PCM using arithmetic right-shift.

        Parameters
        ----------
        samples32 : np.ndarray, dtype=int32
            Input audio samples.

        Returns
        -------
        np.ndarray, dtype=int16
            Converted audio samples.
        """
        samples32 = np.asarray(samples32, dtype=np.int32)
        # Arithmetic right-shift keeps sign for negative values
        return (samples32 >> 16).astype(np.int16)

    def _audio_chunk_to_numpy(self, chunk: AudioChunk) -> np.ndarray:
        """Convert AudioChunk bytes to numpy array as float32."""
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

        # Normalize by the appropriate scale factor for the bit depth
        if chunk.width == 1:
            scale_factor = float(np.iinfo(np.int8).max)  # 2^7
        elif chunk.width == 2:
            scale_factor = float(np.iinfo(np.int16).max)  # 2^15
        elif chunk.width == 4:
            scale_factor = float(np.iinfo(np.int32).max)  # 2^31
        else:
            raise ValueError(f"Unsupported audio width: {chunk.width}")

        audio_array = audio_array.astype(np.float32) / scale_factor

        return audio_array

    def _numpy_to_audio_chunk(
        self, audio_array: np.ndarray, rate: int, width: int, channels: int
    ) -> AudioChunk:
        """Convert numpy array back to AudioChunk."""
        # Determine the numpy dtype and scale factor based on audio width
        dtype: type[np.signedinteger] | type[np.floating]
        if width == 1:
            dtype = np.int8
            max_val = np.iinfo(np.int8).max
            scale_factor = float(np.iinfo(np.int8).max)
        elif width == 2:
            dtype = np.int16
            max_val = np.iinfo(np.int16).max
            scale_factor = float(np.iinfo(np.int16).max)
        elif width == 4:
            dtype = np.int32
            max_val = np.iinfo(np.int32).max
            scale_factor = float(np.iinfo(np.int32).max)
        else:
            raise ValueError(f"Unsupported audio width: {width}")

        # If we have normalized floats, scale back to integer range
        if np.issubdtype(audio_array.dtype, np.floating):
            audio_array = audio_array * scale_factor
            audio_array = np.rint(audio_array).clip(-max_val, max_val)

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
            # Store input chunk properties for safe flush
            self._input_width = chunk.width
            self._input_channels = chunk.channels

            if self._resample_channels is None:
                self._resample_channels = chunk.channels
                logger.info(f"Inferred resample_channels from input: {self._resample_channels}")

            if self._resample_width is None:
                self._resample_width = chunk.width
                logger.info(f"Inferred resample_width from input: {self._resample_width}")

            self._initialized = True

    def _stereo_to_mono(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono by averaging channels.

        Parameters
        ----------
        audio_array : np.ndarray
            Input stereo audio array with shape (samples, 2)

        Returns
        -------
        np.ndarray
            Mono audio array with shape (samples,)
        """
        if audio_array.ndim == 1:
            return audio_array  # Already mono

        if audio_array.shape[-1] != 2:
            raise ValueError(
                f"Expected stereo input (2 channels), got {audio_array.shape[-1]} channels"
            )

        # Average the two channels
        return np.mean(audio_array, axis=-1)

    def _mono_to_stereo(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Convert mono audio to stereo by duplicating the channel.

        Parameters
        ----------
        audio_array : np.ndarray
            Input mono audio array with shape (samples,)

        Returns
        -------
        np.ndarray
            Stereo audio array with shape (samples, 2)
        """
        if audio_array.ndim == 2 and audio_array.shape[-1] == 2:
            return audio_array  # Already stereo

        if audio_array.ndim == 2 and audio_array.shape[-1] != 1:
            raise ValueError(
                f"Expected mono input (1 channel), got {audio_array.shape[-1]} channels"
            )

        # Ensure we have a 1D array for mono
        if audio_array.ndim == 2:
            audio_array = audio_array.squeeze(-1)

        # Duplicate the channel to create stereo
        return np.repeat(audio_array[:, np.newaxis], 2, axis=-1)

    async def process(self, input_stream: AudioStream) -> AudioStream:
        """Process a stream of audio chunks with optimal resampling state management."""
        async for chunk in input_stream:
            # Initialize unspecified parameters from first chunk
            self._initialize_from_first_chunk(chunk)

            # Now we know all parameters are set
            assert self._resample_channels is not None
            assert self._resample_width is not None

            # Check for unsupported audio widths
            if chunk.width not in [2, 4] or self._resample_width not in [2, 4]:
                raise NotImplementedError("Only 16-bit and 32-bit audio processing is supported")

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
                    # Stereo to mono
                    audio_array = self._stereo_to_mono(audio_array)
                elif chunk.channels == 1 and self._resample_channels == 2:
                    # Mono to stereo
                    audio_array = self._mono_to_stereo(audio_array)
                else:
                    raise ValueError(
                        f"Unsupported channel conversion: {chunk.channels} -> {self._resample_channels}. "
                        f"Only mono ↔ stereo conversions are supported."
                    )

            # Resample if needed (handles both mono and multi-channel audio)
            if chunk.rate != self._resample_rate:
                # Create or reuse ResampleStream to maintain filter state between chunks
                if self._resample_stream is None:
                    self._resample_stream = soxr.ResampleStream(
                        chunk.rate,
                        self._resample_rate,
                        self._resample_channels,
                        dtype="float32",
                        quality=self._quality,
                    )
                resampled = self._resample_stream.resample_chunk(audio_array, last=False)

                # Yield control to event loop after heavy SoXR processing
                await asyncio.sleep(0)
            else:
                resampled = audio_array

            # Handle bit depth conversion if needed
            if chunk.width != self._resample_width:
                if chunk.width == 4 and self._resample_width == 2:
                    # Convert 32-bit to 16-bit using arithmetic right-shift
                    # Scale back from normalized float32 to int32 range first
                    if np.issubdtype(resampled.dtype, np.floating):
                        int32_scale = float(np.iinfo(np.int32).max)
                        int32_min = np.iinfo(np.int32).min
                        int32_max = np.iinfo(np.int32).max
                        resampled = resampled * int32_scale
                        resampled = np.rint(resampled).clip(int32_min, int32_max).astype(np.int32)
                    resampled_int = self._int32_to_int16(resampled)
                elif chunk.width == 2 and self._resample_width == 4:
                    # Convert 16-bit to 32-bit by scaling up
                    # Scale back from normalized float32 to int16 range first, then scale to int32
                    if np.issubdtype(resampled.dtype, np.floating):
                        int16_scale = float(np.iinfo(np.int16).max)
                        int16_min = np.iinfo(np.int16).min
                        int16_max = np.iinfo(np.int16).max
                        resampled = resampled * int16_scale
                        resampled = np.rint(resampled).clip(int16_min, int16_max).astype(np.int16)
                    resampled_int = resampled.astype(np.int32) << 16
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

        # Flush any remaining samples from the ResampleStream at end of stream
        # Only flush if we processed at least one chunk (initialized) and used resampling
        if self._resample_stream is not None and self._initialized:
            # At this point, _resample_channels is guaranteed to be not None
            assert self._resample_channels is not None

            # Create empty array with correct shape for the number of channels
            if self._resample_channels == 1:
                empty_array = np.array([], dtype=np.float32)
            else:
                # For multi-channel, create empty 2D array with correct shape
                empty_array = np.empty((0, self._resample_channels), dtype=np.float32)

            # Flush remaining samples by calling with empty array and last=True
            remaining_samples = self._resample_stream.resample_chunk(empty_array, last=True)

            if len(remaining_samples) > 0:
                # Handle bit depth conversion on remaining samples if needed
                if self._resample_width != self._input_width:  # Use stored input chunk's width
                    if self._input_width == 4 and self._resample_width == 2:
                        # Convert 32-bit to 16-bit using arithmetic right-shift
                        if np.issubdtype(remaining_samples.dtype, np.floating):
                            int32_scale = float(np.iinfo(np.int32).max)
                            int32_min = np.iinfo(np.int32).min
                            int32_max = np.iinfo(np.int32).max
                            remaining_samples = remaining_samples * int32_scale
                            remaining_samples = (
                                np.rint(remaining_samples)
                                .clip(int32_min, int32_max)
                                .astype(np.int32)
                            )
                        remaining_samples = self._int32_to_int16(remaining_samples)
                    elif self._input_width == 2 and self._resample_width == 4:
                        # Convert 16-bit to 32-bit by scaling up
                        if np.issubdtype(remaining_samples.dtype, np.floating):
                            int16_scale = float(np.iinfo(np.int16).max)
                            int16_min = np.iinfo(np.int16).min
                            int16_max = np.iinfo(np.int16).max
                            remaining_samples = remaining_samples * int16_scale
                            remaining_samples = (
                                np.rint(remaining_samples)
                                .clip(int16_min, int16_max)
                                .astype(np.int16)
                            )
                        remaining_samples = remaining_samples.astype(np.int32) << 16

                # Create final AudioChunk with remaining samples
                # At this point, _resample_width and _resample_channels are guaranteed to be not None
                assert self._resample_width is not None
                assert self._resample_channels is not None
                final_chunk = self._numpy_to_audio_chunk(
                    remaining_samples,
                    self._resample_rate,
                    self._resample_width,
                    self._resample_channels,
                )
                yield final_chunk

    async def open(self):
        self._initialized = False
        self._resample_stream = None
        self._input_width = None
        self._input_channels = None

    async def close(self):
        self._initialized = False
        if self._resample_stream is not None:
            self._resample_stream = None
        self._input_width = None
        self._input_channels = None


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

    async def process_chunk(self, chunk: AudioChunk) -> AsyncIterator[AudioChunk]:
        """Optimized single-chunk processing method."""
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

        # Don't yield remaining buffer for single chunk processing
        # The remaining buffer will be handled by the streaming process method

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

        # Yield any remaining audio in buffer at end of stream
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
