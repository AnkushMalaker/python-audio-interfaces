from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Protocol, Type

from wyoming.audio import AudioChunk

from easy_audio_interfaces.types.common import AudioStream


class AudioSource(AudioStream, Protocol):
    """Abstract source class that can be used to read from a file or stream."""

    async def read(self) -> Optional[AudioChunk]:
        """Read the next audio segment. Return None if no more data."""
        ...

    async def open(self): ...

    async def close(self): ...

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
    def sample_rate(self) -> int | float: ...

    @property
    def channels(self) -> int: ...

    def __aiter__(self) -> AsyncIterator[AudioChunk]:
        return self.iter_frames()

    async def iter_frames(self) -> AsyncIterator[AudioChunk]:
        """Iterate over audio frames."""
        while True:
            frame = await self.read()
            if frame is None:
                break
            yield frame


class AudioSinkProtocol(Protocol):
    """Protocol for audio sinks - supports structural subtyping and duck typing.

    Use this for type annotations to preserve Protocol benefits.
    Implementations can satisfy this protocol without inheritance.
    """

    @property
    def sample_rate(self) -> int | float: ...

    @property
    def channels(self) -> int: ...

    @property
    def sample_width(self) -> int: ...

    async def write(self, data: AudioChunk) -> None: ...

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def __aenter__(self) -> "AudioSinkProtocol": ...

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ): ...

    async def write_from(self, input_stream: AudioStream): ...


class BaseAudioSink(ABC):
    """Base implementation for audio sinks with automatic validation.

    Inherit from this class to get automatic chunk validation.
    Implements AudioSinkProtocol and provides template method pattern.
    Override _write_impl() instead of write() to get validation for free.
    """

    @property
    @abstractmethod
    def sample_rate(self) -> int | float:
        """The expected sample rate for this sink."""
        ...

    @property
    @abstractmethod
    def channels(self) -> int:
        """The expected number of channels for this sink."""
        ...

    @property
    @abstractmethod
    def sample_width(self) -> int:
        """The expected sample width in bytes for this sink."""
        ...

    def _validate_chunk(self, chunk: AudioChunk) -> None:
        """Validate that chunk format matches sink requirements.

        This is extremely fast - just 3 integer comparisons.
        """
        if chunk.rate != self.sample_rate:
            raise ValueError(f"Sample rate mismatch: {chunk.rate} != {self.sample_rate}")
        if chunk.channels != self.channels:
            raise ValueError(f"Channel mismatch: {chunk.channels} != {self.channels}")
        if chunk.width != self.sample_width:
            raise ValueError(f"Sample width mismatch: {chunk.width} != {self.sample_width}")

    async def write(self, data: AudioChunk) -> None:
        """Write audio data with automatic validation.

        This method validates the chunk format and then calls _write_impl().
        Override _write_impl() to provide custom write behavior.
        """
        self._validate_chunk(data)
        await self._write_impl(data)

    @abstractmethod
    async def _write_impl(self, data: AudioChunk) -> None:
        """Implementation-specific write logic.

        This method is called after chunk validation passes.
        Override this method to implement custom write behavior.
        """
        ...

    @abstractmethod
    async def open(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    async def __aenter__(self) -> "BaseAudioSink":
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


# Type alias for backwards compatibility and cleaner imports
AudioSink = AudioSinkProtocol


class ProcessingBlock(Protocol):
    """Abstract processing block that can be used to process audio data."""

    def process(self, input_stream: AudioStream) -> AudioStream: ...

    async def process_chunk(self, chunk: AudioChunk) -> AsyncIterator[AudioChunk]:
        """Convenience method for processing a single AudioChunk.

        Default implementation falls back to .process() method.
        Blocks that care about performance can override this with a real fast-path.
        """

        async def _single() -> AsyncIterator[AudioChunk]:
            yield chunk

        async for out in self.process(_single()):
            yield out

    async def open(self): ...

    async def close(self): ...

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
