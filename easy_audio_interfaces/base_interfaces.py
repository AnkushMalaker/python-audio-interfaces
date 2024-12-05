from typing import AsyncIterable, AsyncIterator, Optional, Protocol, Type

from pydub import AudioSegment


class AudioSource(Protocol):
    """Abstract source class that can be used to read from a file or stream."""

    async def read(self) -> Optional[AudioSegment]:
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

    def __aiter__(self) -> AsyncIterator[AudioSegment]:
        return self.iter_frames()

    async def iter_frames(self) -> AsyncIterator[AudioSegment]:
        """Iterate over audio frames."""
        while True:
            frame = await self.read()
            if frame is None:
                break
            yield frame


class AudioSink(Protocol):
    """Abstract sink class that can be used to write to a file or stream."""

    async def write(self, data: AudioSegment):
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

    async def write_from(self, input_stream: AsyncIterable[AudioSegment]):
        async for chunk in input_stream:
            await self.write(chunk)


class ProcessingBlock(Protocol):
    """Abstract processing block that can be used to process audio data."""

    def process(self, input_stream: AsyncIterable[AudioSegment]) -> AsyncIterable[AudioSegment]:
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
