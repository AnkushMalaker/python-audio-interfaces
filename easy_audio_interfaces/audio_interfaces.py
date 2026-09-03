import asyncio
import logging
from io import BytesIO
from typing import AsyncGenerator

import numpy as np
from numpy.typing import NDArray

from easy_audio_interfaces.base_interfaces import (
    AudioSink,
    AudioSource,
    ChunkedWavAudioSource,
    ProcessingBlock,
)
from easy_audio_interfaces.processing import SampleRateConverter
from easy_audio_interfaces.streaming import WaveFileStreamer
from easy_audio_interfaces.types.audio import AudioChunk, AudioFormat
from easy_audio_interfaces.types.common import AudioStream

logger = logging.getLogger(__name__)


# TODO
class StreamFromCommand(AudioSource):
    def __init__(self, command: str):
        self._command = command

    async def open(self): ...

    async def close(self): ...


class SourceFromBytesIO(ChunkedWavAudioSource):
    def __init__(
        self,
        bytes_io: BytesIO,
        *,
        chunk_size_ms: int | None = None,
        chunk_size_samples: int | None = None,
    ):
        super().__init__(chunk_size_ms=chunk_size_ms, chunk_size_samples=chunk_size_samples)
        self._bytes_io = bytes_io

    def _create_wave_streamer(self) -> WaveFileStreamer:
        return WaveFileStreamer(self._bytes_io)

    async def open(self):
        await super().open()
        logger.info(
            f"Opened BytesIO stream, Sample rate: {self.sample_rate}, Channels: {self.channels}"
        )

    async def close(self):
        await super().close()
        logger.info("Closed BytesIO stream")


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
        - Sample rate conversion using  SoXR library
        - Channel conversion: mono ↔ stereo via mean (instead of random pick)
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
        self._configured_channels = resample_channels
        self._configured_width = resample_width
        self._resample_channels = resample_channels
        self._resample_width = resample_width
        self._quality = quality

        self._input_format: AudioFormat | None = None
        self._sample_rate_converter = SampleRateConverter(resample_rate, quality=quality)

    @property
    def sample_rate(self) -> int:
        return self._resample_rate

    def _int32_to_int16(self, samples32: NDArray[np.int32]) -> NDArray[np.int16]:
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

    def _normalized_float_to_int(
        self,
        samples: NDArray[np.floating] | NDArray[np.signedinteger],
        dtype: type[np.signedinteger],
    ) -> NDArray[np.signedinteger]:
        """Convert normalized floats without overflowing at integer boundaries."""

        limits = np.iinfo(dtype)
        scaled = np.asarray(samples, dtype=np.float64) * float(limits.max)
        return np.rint(scaled).clip(limits.min, limits.max).astype(dtype)

    def _audio_chunk_to_numpy_float(self, chunk: AudioChunk) -> NDArray[np.floating]:
        """Convert AudioChunk bytes to numpy array as float32."""
        dtype: type[np.signedinteger]
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
        audio_array: np.ndarray = np.frombuffer(chunk.audio, dtype=dtype)

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
        self,
        audio_array: NDArray[np.floating] | NDArray[np.signedinteger],
        rate: int,
        width: int,
        channels: int,
    ) -> AudioChunk:
        """Convert numpy array back to AudioChunk."""
        # Determine the numpy dtype and scale factor based on audio width
        dtype: type[np.signedinteger]
        if width == 1:
            dtype = np.int8
        elif width == 2:
            dtype = np.int16
        elif width == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported audio width: {width}")

        # If we have normalized floats, scale back to integer range
        if np.issubdtype(audio_array.dtype, np.floating):
            processed_array = self._normalized_float_to_int(audio_array, dtype)
        else:
            # Ensure the array is the correct dtype
            processed_array = audio_array.astype(dtype)

        # Convert back to bytes
        audio_bytes = processed_array.tobytes()

        return AudioChunk(
            audio=audio_bytes,
            rate=rate,
            width=width,
            channels=channels,
        )

    def _initialize_from_first_chunk(self, chunk: AudioChunk) -> None:
        """Infer output fields once and fix the input format for the stream."""

        if self._input_format is not None:
            if chunk.format != self._input_format:
                raise ValueError(
                    f"input format changed from {self._input_format} to {chunk.format}; "
                    "reset before starting a new stream"
                )
            return

        self._input_format = chunk.format
        if self._resample_channels is None:
            self._resample_channels = chunk.channels
        if self._resample_width is None:
            self._resample_width = chunk.width

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

    def _convert_output_format(self, chunk: AudioChunk) -> AudioChunk:
        """Apply the block's legacy channel and width conversions after resampling."""

        assert self._resample_channels is not None
        assert self._resample_width is not None

        if chunk.channels == self._resample_channels and chunk.width == self._resample_width:
            return chunk

        audio_array = self._audio_chunk_to_numpy_float(chunk)
        if chunk.channels != self._resample_channels:
            if chunk.channels == 2 and self._resample_channels == 1:
                audio_array = self._stereo_to_mono(audio_array)
            elif chunk.channels == 1 and self._resample_channels == 2:
                audio_array = self._mono_to_stereo(audio_array)
            else:
                raise ValueError(
                    f"Unsupported channel conversion: {chunk.channels} -> "
                    f"{self._resample_channels}. Only mono ↔ stereo conversions are supported."
                )

        converted_width: NDArray[np.floating] | NDArray[np.signedinteger]
        if chunk.width == self._resample_width:
            converted_width = audio_array
        elif chunk.width == 4 and self._resample_width == 2:
            samples32 = self._normalized_float_to_int(audio_array, np.int32)
            converted_width = self._int32_to_int16(samples32.astype(np.int32))
        elif chunk.width == 2 and self._resample_width == 4:
            samples16 = self._normalized_float_to_int(audio_array, np.int16)
            converted_width = samples16.astype(np.int32) << 16
        else:
            raise ValueError(
                f"Unsupported bit depth conversion: {chunk.width} -> {self._resample_width}"
            )

        return self._numpy_to_audio_chunk(
            converted_width,
            self._resample_rate,
            self._resample_width,
            self._resample_channels,
        )

    def _validate_supported_widths(self, chunk: AudioChunk) -> None:
        assert self._resample_width is not None
        if chunk.width not in {2, 4} or self._resample_width not in {2, 4}:
            raise NotImplementedError("Only 16-bit and 32-bit audio processing is supported")
        assert self._resample_channels is not None
        if chunk.channels != self._resample_channels and {
            chunk.channels,
            self._resample_channels,
        } != {1, 2}:
            raise ValueError(
                f"Unsupported channel conversion: {chunk.channels} -> "
                f"{self._resample_channels}. Only mono ↔ stereo conversions are supported."
            )

    async def process(self, input_stream: AudioStream) -> AudioStream:
        """Convert an async stream and finalize its shared resampling state once."""

        async for chunk in input_stream:
            async for output in self.process_chunk(chunk):
                yield output
        async for output in self.process_chunk_last():
            yield output

    async def process_chunk_last(self) -> AsyncGenerator[AudioChunk, None]:
        """
        Flush any remaining samples from the ResampleStream buffer.

        This method should be called at the end of processing to ensure
        all buffered samples are yielded, preventing sample loss.

        Yields
        ------
        AudioChunk
            Any remaining buffered audio chunks after flushing
        """
        remaining = self._sample_rate_converter.finish()
        if remaining is not None:
            yield self._convert_output_format(remaining)

    async def process_chunk(self, chunk: AudioChunk) -> AsyncGenerator[AudioChunk, None]:
        self._initialize_from_first_chunk(chunk)
        self._validate_supported_widths(chunk)

        converted = self._sample_rate_converter.push(chunk)
        if chunk.rate != self._resample_rate:
            await asyncio.sleep(0)
        if converted is not None:
            yield self._convert_output_format(converted)

    async def open(self):
        self.reset()

    async def close(self):
        self.reset()

    def reset(self):
        self._input_format = None
        self._resample_channels = self._configured_channels
        self._resample_width = self._configured_width
        self._sample_rate_converter.reset()


class RechunkingBlock(ProcessingBlock):
    def __init__(self, *, chunk_size_ms: int | None = None, chunk_size_samples: int | None = None):
        if chunk_size_ms is None and chunk_size_samples is None:
            raise ValueError("Either chunk_size_ms or chunk_size_samples must be provided")
        if chunk_size_ms is not None and chunk_size_samples is not None:
            raise ValueError("Only one of chunk_size_ms or chunk_size_samples can be provided")
        if chunk_size_ms is not None and chunk_size_ms <= 0:
            raise ValueError("chunk_size_ms must be greater than zero")
        if chunk_size_samples is not None and chunk_size_samples <= 0:
            raise ValueError("chunk_size_samples must be greater than zero")

        self._chunk_size_ms = chunk_size_ms
        self._chunk_size_samples = chunk_size_samples
        self._buffer = b""
        self._audio_format: AudioFormat | None = None

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

        if target_samples <= 0:
            raise ValueError(
                "chunk_size_ms is too small to contain one sample at " f"{audio_format.rate} Hz"
            )

        return self._samples_to_bytes(target_samples, audio_format.width, audio_format.channels)

    async def process_chunk(self, chunk: AudioChunk) -> AsyncGenerator[AudioChunk, None]:
        # Store audio format from the first chunk
        if self._audio_format is None:
            self._audio_format = chunk.format
        elif chunk.format != self._audio_format:
            raise ValueError(f"Audio format changed from {self._audio_format} to {chunk.format}")

        target_chunk_size = self._get_target_chunk_size_bytes(chunk)

        # Add new audio data to buffer
        self._buffer += chunk.audio

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

    async def process(self, input_stream: AudioStream) -> AudioStream:
        """Unified processing method that works with both ms and samples by converting to bytes."""
        async for chunk in input_stream:
            async for out in self.process_chunk(chunk):
                yield out

        # Yield any remaining audio in buffer at end of stream
        if len(self._buffer) > 0 and self._audio_format is not None:
            yield AudioChunk(
                audio=self._buffer,
                rate=self._audio_format.rate,
                width=self._audio_format.width,
                channels=self._audio_format.channels,
            )
            self._buffer = b""

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
