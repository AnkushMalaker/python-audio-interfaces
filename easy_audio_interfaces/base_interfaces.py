from typing import AsyncGenerator, AsyncIterable, Optional, Protocol, Type

from easy_audio_interfaces.types.audio import NumpyFrame


class AudioSource(AsyncIterable, Protocol):
    """Abstract source class that can be used to read from a file or stream."""

    async def read(self) -> NumpyFrame:
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
    def sample_rate(self) -> int:
        ...

    @property
    def channels(self) -> int:
        ...

    def __aiter__(self) -> AsyncGenerator[NumpyFrame, None]:
        return self.iter_frames()

    async def iter_frames(self) -> AsyncGenerator[NumpyFrame, None]:
        async for frame in self:
            yield frame


class AudioSink(AsyncIterable, Protocol):
    """Abstract sink class that can be used to write to a file or stream."""

    async def write(self, data: NumpyFrame):
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

    async def write_from(self, input_stream: AsyncIterable[NumpyFrame]):
        async for chunk in input_stream:
            await self.write(chunk)


class ProcessingBlock(AsyncIterable, Protocol):
    """Abstract processing block class that can be used to process audio data."""

    async def process(self, input_stream: AsyncIterable[NumpyFrame]) -> AsyncIterable[NumpyFrame]:
        ...

    async def open(self):
        pass

    async def close(self):
        pass

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
