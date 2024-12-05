import logging
from typing import cast

from pydub import AudioSegment

from easy_audio_interfaces.base_interfaces import (
    AudioSink,
    AudioSource,
    ProcessingBlock,
)
from easy_audio_interfaces.types.common import AudioStream

logger = logging.getLogger(__name__)


class ResamplingBlock(ProcessingBlock):
    def __init__(
        self,
        resample_rate: int,
    ):
        self._resample_rate = resample_rate

    @property
    def sample_rate(self) -> int:
        return self._resample_rate

    # FIXME: Why the type error?
    async def process(self, input_stream: AudioStream) -> AudioStream:
        async for frame in input_stream:
            # Use AudioSegment's built-in frame rate conversion
            frame = cast(AudioSegment, frame.set_frame_rate(self._resample_rate))
            yield frame

    async def open(self):
        ...

    async def close(self):
        ...


class RechunkingBlock(ProcessingBlock):
    def __init__(self, *, chunk_size_ms: int | None = None, chunk_size_samples: int | None = None):
        if chunk_size_ms is None and chunk_size_samples is None:
            raise ValueError("Either chunk_size_ms or chunk_size_samples must be provided")
        if chunk_size_ms is not None and chunk_size_samples is not None:
            raise ValueError("Only one of chunk_size_ms or chunk_size_samples can be provided")

        self._chunk_size_ms = chunk_size_ms
        self._chunk_size_samples = chunk_size_samples
        self._buffer: AudioSegment = AudioSegment.silent(duration=0)

    async def _process_chunk_ms(self, input_stream: AudioStream) -> AudioStream:
        async for frame in input_stream:
            self._buffer += frame
            assert self._chunk_size_ms is not None
            chunk_size = int(self._chunk_size_ms)

            while len(self._buffer) >= chunk_size:
                chunk: AudioSegment = cast(AudioSegment, self._buffer[:chunk_size])
                self._buffer = cast(AudioSegment, self._buffer[chunk_size:])
                yield chunk

        if len(self._buffer) > 0:
            yield self._buffer
            self._buffer = AudioSegment.silent(duration=0)

    async def _process_chunk_samples(self, input_stream: AudioStream) -> AudioStream:
        assert self._chunk_size_samples is not None
        chunk_size = self._chunk_size_samples
        async for frame in input_stream:
            self._buffer += frame
            while int(self._buffer.frame_count()) >= chunk_size:
                chunk_data = self._buffer.get_array_of_samples()
                num_samples = len(chunk_data) // self._buffer.channels

                # Construct and yield the chunks
                for idx in range(0, num_samples, chunk_size):
                    chunk: AudioSegment = AudioSegment(
                        data=chunk_data[idx : idx + chunk_size],
                        sample_width=self._buffer.frame_width,
                        frame_rate=self._buffer.frame_rate,
                        channels=self._buffer.channels,
                    )
                    if chunk.frame_count() < chunk_size:
                        self._buffer = chunk
                    else:
                        yield chunk

        if len(self._buffer) > 0:
            yield self._buffer
            self._buffer = AudioSegment.silent(duration=0)

    def process(self, input_stream: AudioStream) -> AudioStream:
        if self._chunk_size_ms is not None:
            return self._process_chunk_ms(input_stream)
        else:
            return self._process_chunk_samples(input_stream)

    async def open(self):
        ...

    async def close(self):
        ...


__all__ = [
    "AudioSource",
    "AudioSink",
    "ResamplingBlock",
    "RechunkingBlock",
    "ProcessingBlock",
]
