import logging
from pathlib import Path
from typing import AsyncGenerator, AsyncIterable, Iterable, Optional, Type

from pydub import AudioSegment

from easy_audio_interfaces.base_interfaces import AudioSink, AudioSource
from easy_audio_interfaces.types.common import PathLike

logger = logging.getLogger(__name__)


class LocalFileStreamer(AudioSource):
    def __init__(
        self,
        file_path: PathLike,
    ):
        self._file_path = Path(file_path)
        self._audio_segment: Optional[AudioSegment] = None
        self._current_position: int = 0

    @property
    def sample_rate(self) -> int:
        return self._audio_segment.frame_rate if self._audio_segment else 0

    @property
    def channels(self) -> int:
        return self._audio_segment.channels if self._audio_segment else 0

    async def open(self):
        self._audio_segment = AudioSegment.from_file(self._file_path)
        if self._audio_segment is None:
            raise RuntimeError(f"Failed to open file: {self._file_path}")
        logger.info(
            f"Opened file: {self._file_path}, Sample rate: {self._audio_segment.frame_rate}, Channels: {self._audio_segment.channels}"
        )

    async def read(self) -> AudioSegment:
        if self._audio_segment is None:
            raise RuntimeError("File is not open. Call 'open()' first.")

        if self._audio_segment.frame_count() == 0:
            return AudioSegment.silent(duration=0)

        return self._audio_segment

    async def close(self):
        if self._audio_segment:
            self._audio_segment.export(self._file_path, format="wav")
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

    async def iter_frames(self) -> AsyncGenerator[AudioSegment, None]:
        while True:
            frame = await self.read()
            if frame.frame_count() == 0:
                break
            yield frame


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
        self._audio_segment: Optional[AudioSegment] = None

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
        self._audio_segment = AudioSegment.silent(duration=0)
        logger.info(f"Opened file for writing: {self._file_path}")

    async def write(self, data: AudioSegment):
        if self._audio_segment is None:
            raise RuntimeError("File is not open. Call 'open()' first.")
        self._audio_segment += data
        logger.debug(f"Wrote {len(data)} bytes to {self._file_path}")

    async def write_from(self, input_stream: AsyncIterable[AudioSegment] | Iterable[AudioSegment]):
        total_frames = 0
        total_bytes = 0
        if isinstance(input_stream, AsyncIterable):
            async for chunk in input_stream:
                await self.write(chunk)
                total_frames += 1
                total_bytes += len(chunk)
        else:
            for chunk in input_stream:
                await self.write(chunk)
                total_frames += 1
                total_bytes += len(chunk)
        logger.info(
            f"Finished writing {total_frames} frames ({total_bytes} bytes) to {self._file_path}"
        )

    async def close(self):
        if self._audio_segment:
            self._audio_segment.export(self._file_path, format="wav")
            self._audio_segment = None
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
