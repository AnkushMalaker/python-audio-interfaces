import logging
import wave
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, AsyncIterable, Callable, Iterable, Optional, Type

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


class RollingFileSink(AudioSink):
    def __init__(
        self,
        directory: PathLike,
        prefix: str,
        segment_duration_seconds: int | float,
        sample_rate: int | float,
        channels: int,
        sample_width: int = 2,  # Default to 16-bit audio
    ):
        self._directory = Path(directory)
        self._prefix = prefix
        self._segment_duration_seconds = segment_duration_seconds
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width

        # Calculate the maximum samples per segment
        self._max_samples_per_segment = int(segment_duration_seconds * sample_rate)

        # Track current file state
        self._current_sink: Optional[LocalFileSink] = None
        self._current_file_path: Optional[Path] = None
        self._current_samples_written: int = 0
        self._file_counter: int = 0
        self.generate_filename: Callable[[], str] = self._generate_filename

    @property
    def sample_rate(self) -> int | float:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def _generate_filename(self) -> str:
        """Generate a timestamped filename with counter."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
        return f"{self._prefix}_{timestamp}_{self._file_counter:03d}.wav"

    async def _roll_file(self):
        """Close current file and start a new one."""
        if self._current_sink:
            await self._current_sink.close()
            logger.info(f"Closed rolled file: {self._current_file_path}")

        # Generate new file path
        filename = self.generate_filename()
        self._current_file_path = self._directory / filename

        # Create new LocalFileSink instance
        self._current_sink = LocalFileSink(
            file_path=self._current_file_path,
            sample_rate=self._sample_rate,
            channels=self._channels,
            sample_width=self._sample_width,
        )
        await self._current_sink.open()

        self._current_samples_written = 0
        self._file_counter += 1

        logger.info(f"Started new rolled file: {self._current_file_path}")

    async def open(self):
        logger.debug(f"Opening rolling file sink in directory: {self._directory}")

        # Create directory if it doesn't exist
        if not self._directory.exists():
            self._directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {self._directory}")

        if not self._directory.is_dir():
            raise RuntimeError(f"Path exists but is not a directory: {self._directory}")

        # Start the first file
        await self._roll_file()
        logger.info(f"Opened rolling file sink in directory: {self._directory}")

    async def write(self, data: AudioChunk):
        if self._current_sink is None:
            raise RuntimeError("File sink is not open. Call 'open()' first.")

        # Calculate how many samples this chunk contains
        chunk_samples = len(data.audio) // (self._sample_width * self._channels)

        # Check if adding this chunk would exceed our segment limit
        if self._current_samples_written + chunk_samples > self._max_samples_per_segment:
            # Roll to a new file before writing this chunk
            await self._roll_file()

        # Write the entire chunk to current file
        await self._current_sink.write(data)
        self._current_samples_written += chunk_samples

        logger.debug(
            f"Wrote {len(data.audio)} bytes ({chunk_samples} samples) to {self._current_file_path}"
        )

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
            f"Finished writing {total_frames} frames ({total_bytes} bytes) across {self._file_counter} files to {self._directory}"
        )

    async def close(self):
        if self._current_sink:
            await self._current_sink.close()
            self._current_sink = None
            logger.info(f"Closed final rolled file: {self._current_file_path}")
        logger.info(
            f"Closed rolling file sink. Created {self._file_counter} files in {self._directory}"
        )

    async def __aenter__(self) -> "RollingFileSink":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ):
        await self.close()
