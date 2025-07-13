import logging
import wave
from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator, AsyncIterable, Callable, Iterable, Optional, Type

from wyoming.audio import AudioChunk

from easy_audio_interfaces.base_interfaces import AudioSink, ChunkedWavAudioSource
from easy_audio_interfaces.streaming import WaveFileStreamer
from easy_audio_interfaces.types.common import PathLike

logger = logging.getLogger(__name__)


class LocalFileStreamer(ChunkedWavAudioSource):
    def __init__(
        self,
        file_path: PathLike,
        *,
        chunk_size_ms: int | None = None,
        chunk_size_samples: int | None = None,
    ):
        super().__init__(chunk_size_ms=chunk_size_ms, chunk_size_samples=chunk_size_samples)
        self._file_path = Path(file_path)

    def _create_wave_streamer(self) -> WaveFileStreamer:
        if not self._file_path.exists():
            raise FileNotFoundError(f"File not found: {self._file_path}")
        return WaveFileStreamer(self._file_path)

    async def open(self):
        await super().open()
        logger.info(
            f"Opened file: {self._file_path}, Sample rate: {self.sample_rate}, Channels: {self.channels}"
        )

    async def read(self) -> AudioChunk:
        result = await super().read()
        if result is None:
            raise StopAsyncIteration
        return result

    async def close(self):
        await super().close()
        logger.info(f"Closed file: {self._file_path}")

    async def __aenter__(self) -> "LocalFileStreamer":
        await self.open()
        return self

    async def __aexit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        _exc_value: Optional[BaseException],
        _traceback: Optional[Type[BaseException]],
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

    @property
    def file_path(self) -> Path:
        return self._file_path


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

    @property
    def file_path(self) -> Path:
        return self._file_path

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
        _exc_type: Optional[Type[BaseException]],
        _exc_value: Optional[BaseException],
        _traceback: Optional[Type[BaseException]],
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

        # Calculate the target samples per segment and use rechunking for exact durations
        target_samples_per_segment = int(segment_duration_seconds * sample_rate)

        # Import here to avoid circular imports
        from easy_audio_interfaces.audio_interfaces import RechunkingBlock

        self._rechunker = RechunkingBlock(chunk_size_samples=target_samples_per_segment)

        # Track current file state
        self._current_sink: Optional[LocalFileSink] = None
        self._current_file_path: Optional[Path] = None
        self._file_counter: int = 0
        self.generate_filename: Callable[[], str] = self._generate_filename

    @property
    def sample_rate(self) -> int | float:
        return self._sample_rate

    @property
    def directory(self) -> Path:
        return self._directory

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

        # Initialize the rechunker
        await self._rechunker.open()

        # Start the first file
        await self._roll_file()
        logger.info(f"Opened rolling file sink in directory: {self._directory}")

    async def write(self, data: AudioChunk):
        if self._current_sink is None:
            raise RuntimeError("File sink is not open. Call 'open()' first.")

        # Use the rechunker to get exactly-sized chunks
        async for segment_chunk in self._rechunker.process_chunk(data):
            # Each chunk from the rechunker represents exactly one file segment
            await self._current_sink.write(segment_chunk)
            await self._current_sink.close()

            # Log the completed file
            logger.info(f"Completed file segment: {self._current_file_path}")

            # Start a new file for the next segment
            await self._roll_file()

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
        # Flush any remaining data from the rechunker
        if self._rechunker._buffer and self._current_sink:
            # Process any remaining buffered data
            remaining_chunk = AudioChunk(
                audio=self._rechunker._buffer,
                rate=int(self._sample_rate),
                width=self._sample_width,
                channels=self._channels,
            )
            await self._current_sink.write(remaining_chunk)
            logger.info(f"Wrote final partial segment: {self._current_file_path}")

        if self._current_sink:
            await self._current_sink.close()
            self._current_sink = None
            logger.info(f"Closed final rolled file: {self._current_file_path}")

        # Close the rechunker
        await self._rechunker.close()

        logger.info(
            f"Closed rolling file sink. Created {self._file_counter} files in {self._directory}"
        )

    async def __aenter__(self) -> "RollingFileSink":
        await self.open()
        return self

    async def __aexit__(
        self,
        _exc_type: Optional[Type[BaseException]],
        _exc_value: Optional[BaseException],
        _traceback: Optional[Type[BaseException]],
    ):
        await self.close()
