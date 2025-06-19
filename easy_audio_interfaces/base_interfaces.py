from typing import AsyncIterator, Optional, Protocol, Type

from wyoming.audio import AudioChunk

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

    def process(self, input_stream: AudioStream) -> AudioStream:
        ...

    async def process_chunk(self, chunk: AudioChunk) -> AsyncIterator[AudioChunk]:
        """Convenience method for processing a single AudioChunk.

        Default implementation falls back to .process() method.
        Blocks that care about performance can override this with a real fast-path.
        """

        async def _single() -> AsyncIterator[AudioChunk]:
            yield chunk

        async for out in self.process(_single()):
            yield out

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
