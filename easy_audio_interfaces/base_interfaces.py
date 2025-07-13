from abc import abstractmethod
from typing import AsyncGenerator, AsyncIterator, Optional, Protocol, Type

from wyoming.audio import AudioChunk

from easy_audio_interfaces.streaming import WaveFileStreamer
from easy_audio_interfaces.types.common import AudioStream


class AudioSource(AudioStream, Protocol):
    """Abstract source class that can be used to read from a file or stream."""

    async def read(self) -> Optional[AudioChunk]:
        """Read the next audio segment. Return None if no more data."""
        ...

    async def open(self):
        ...

    async def close(self):
        ...

    async def __aenter__(self) -> "AudioSource":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ):
        await self.close()

    @property
    def sample_rate(self) -> int | float:
        ...

    @property
    def channels(self) -> int:
        ...

    def __aiter__(self) -> AsyncIterator[AudioChunk]:
        return self.iter_frames()

    async def iter_frames(self) -> AsyncIterator[AudioChunk]:
        """Iterate over audio frames."""
        while True:
            frame = await self.read()
            if frame is None:
                break
            yield frame


class ChunkedWavAudioSource(AudioSource):
    """Base class for audio sources that read from wave files in chunks."""

    def __init__(
        self,
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

        self._wave_streamer: Optional[WaveFileStreamer] = None

    @property
    def sample_rate(self) -> int:
        return self._wave_streamer.rate if self._wave_streamer else 0

    @property
    def channels(self) -> int:
        return self._wave_streamer.channels if self._wave_streamer else 0

    @abstractmethod
    def _create_wave_streamer(self) -> WaveFileStreamer:
        ...

    async def open(self):
        self._wave_streamer = self._create_wave_streamer()
        self._wave_streamer.open()

    async def read(self) -> Optional[AudioChunk]:
        if self._wave_streamer is None:
            raise RuntimeError("Stream is not open. Call 'open()' first.")

        if self._wave_streamer.frames_remaining <= 0:
            return None

        try:
            # If we're using millisecond-based chunks
            if self._chunk_size_ms is not None:
                return self._wave_streamer.read_chunk_by_duration(self._chunk_size_ms)
            # If we're using sample-based chunks
            elif self._chunk_size_samples is not None:
                return self._wave_streamer.read_chunk_by_samples(self._chunk_size_samples)
            else:
                # Default to 512 samples
                return self._wave_streamer.read_chunk_by_samples(512)
        except StopIteration:
            return None

    async def close(self):
        if self._wave_streamer:
            self._wave_streamer.close()
            self._wave_streamer = None


class AudioSink(Protocol):
    """Abstract sink class that can be used to write to a file or stream."""

    async def write(self, data: AudioChunk):
        ...

    async def open(self):
        ...

    async def close(self):
        ...

    async def __aenter__(self) -> "AudioSink":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ):
        await self.close()

    async def write_from(self, input_stream: AudioStream):
        async for chunk in input_stream:
            await self.write(chunk)


class ProcessingBlock(Protocol):
    """Abstract processing block that can be used to process audio data."""

    # async def process(self, input_stream: AudioStream) -> AudioStream: ...

    def process_chunk(self, chunk: AudioChunk) -> AsyncGenerator[AudioChunk, None]:
        ...

    async def open(self):
        ...

    async def close(self):
        ...

    async def __aenter__(self) -> "ProcessingBlock":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ):
        await self.close()
