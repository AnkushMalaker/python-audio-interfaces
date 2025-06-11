import logging
import wave
from pathlib import Path
from typing import AsyncGenerator, AsyncIterable, Iterable, Optional, Type

from wyoming.audio import AudioChunk

from easy_audio_interfaces.base_interfaces import AudioSink, AudioSource
from easy_audio_interfaces.types.common import PathLike
from easy_audio_interfaces.utils import audio_chunk_from_file

logger = logging.getLogger(__name__)


class LocalFileStreamer(AudioSource):
    def __init__(
        self,
        file_path: PathLike,
        *,
        chunk_size_ms: int | None = None,
        chunk_size_samples: int | None = None,
    ):
        if chunk_size_ms is None and chunk_size_samples is None:
            raise ValueError("Either chunk_size_ms or chunk_size_samples must be provided.")
        if chunk_size_ms is not None and chunk_size_samples is not None:
            raise ValueError("Only one of chunk_size_ms or chunk_size_samples can be provided.")

        self._chunk_size_ms = chunk_size_ms
        self._chunk_size_samples = chunk_size_samples
        if not chunk_size_ms and not chunk_size_samples:
            self._chunk_size_samples = 512

        self._file_path = Path(file_path)
        self._audio_segment: Optional[AudioChunk] = None

    @property
    def sample_rate(self) -> int:
        return self._audio_segment.rate if self._audio_segment else 0

    @property
    def channels(self) -> int:
        return self._audio_segment.channels if self._audio_segment else 0

    async def open(self):
        # @optimization: Can convert this to an iterator maybe for better efficiency?
        self._audio_segment = audio_chunk_from_file(self._file_path)
        if self._audio_segment is None:
            raise RuntimeError(f"Failed to open file: {self._file_path}")
        logger.info(
            f"Opened file: {self._file_path}, Sample rate: {self._audio_segment.rate}, Channels: {self._audio_segment.channels}"
        )

    async def read(self) -> AudioChunk:
        if self._audio_segment is None:
            raise RuntimeError("File is not open. Call 'open()' first.")

        if self._audio_segment.samples == 0:
            raise StopAsyncIteration

        # If we're using millisecond-based chunks
        if self._chunk_size_ms is not None:
            assert self._audio_segment is not None
            # Calculate bytes per millisecond
            bytes_per_ms = (
                self._audio_segment.rate * self._audio_segment.width * self._audio_segment.channels
            ) // 1000
            chunk_size_bytes = self._chunk_size_ms * bytes_per_ms

            chunk = self._audio_segment.audio[:chunk_size_bytes]
            self._audio_segment = AudioChunk(
                audio=self._audio_segment.audio[chunk_size_bytes:],
                rate=self._audio_segment.rate,
                width=self._audio_segment.width,
                channels=self._audio_segment.channels,
            )
            return AudioChunk(
                audio=chunk,
                rate=self._audio_segment.rate,
                width=self._audio_segment.width,
                channels=self._audio_segment.channels,
            )

        # If we're using sample-based chunks
        if self._chunk_size_samples is not None:
            # Calculate bytes for the number of samples
            chunk_size_bytes = (
                self._chunk_size_samples * self._audio_segment.width * self._audio_segment.channels
            )

            chunk = self._audio_segment.audio[:chunk_size_bytes]
            self._audio_segment = AudioChunk(
                audio=self._audio_segment.audio[chunk_size_bytes:],
                rate=self._audio_segment.rate,
                width=self._audio_segment.width,
                channels=self._audio_segment.channels,
            )
            return AudioChunk(
                audio=chunk,
                rate=self._audio_segment.rate,
                width=self._audio_segment.width,
                channels=self._audio_segment.channels,
            )

        raise RuntimeError(
            "No chunk size provided. This shouldn't happen. We should default to 512 samples."
        )

    async def close(self):
        if self._audio_segment:
            self._audio_segment = None
        logger.info(f"Closed file: {self._file_path}")

    async def __aenter__(self) -> "LocalFileStreamer":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ):
        await self.close()

    async def iter_frames(self) -> AsyncGenerator[AudioChunk, None]:
        while True:
            try:
                frame = await self.read()
                # Do we need to check for frame.samples == 0?
                yield frame
            except StopAsyncIteration:
                break


class LocalFileSink(AudioSink):
    def __init__(
        self,
        file_path: PathLike,
        sample_rate: int | float,
        channels: int,
        sample_width: int = 2,  # Default to 16-bit audio
    ):
        self._file_path = Path(file_path)
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._file_handle: Optional[wave.Wave_write] = None

    @property
    def sample_rate(self) -> int | float:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        logger.debug(f"Opening file for writing: {self._file_path}")
        if not self._file_path.parent.exists():
            raise RuntimeError(f"Parent directory does not exist: {self._file_path.parent}")

        self._file_handle = wave.open(str(self._file_path), "wb")
        self._file_handle.setnchannels(self._channels)
        self._file_handle.setsampwidth(self._sample_width)
        self._file_handle.setframerate(self._sample_rate)
        logger.info(f"Opened file for writing: {self._file_path}")

    async def write(self, data: AudioChunk):
        if self._file_handle is None:
            raise RuntimeError("File is not open. Call 'open()' first.")
        self._file_handle.writeframes(data.audio)
        logger.debug(f"Wrote {len(data.audio)} bytes to {self._file_path}.")

    async def write_from(self, input_stream: AsyncIterable[AudioChunk] | Iterable[AudioChunk]):
        total_frames = 0
        total_bytes = 0
        if isinstance(input_stream, AsyncIterable):
            async for chunk in input_stream:
                await self.write(chunk)
                total_frames += 1
                total_bytes += len(chunk.audio)
        else:
            for chunk in input_stream:
                await self.write(chunk)
                total_frames += 1
                total_bytes += len(chunk.audio)
        logger.info(
            f"Finished writing {total_frames} frames ({total_bytes} bytes) to {self._file_path}"
        )

    async def close(self):
        if self._file_handle:
            self._file_handle.close()
            self._file_handle = None
        logger.info(f"Closed file: {self._file_path}")

    async def __aenter__(self) -> "LocalFileSink":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ):
        await self.close()

    async def __aiter__(self):
        # This method should yield frames if needed
        # If not needed, you can make it an empty async generator
        yield
