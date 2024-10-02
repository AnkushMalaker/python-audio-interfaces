import asyncio
import logging
from asyncio import StreamReader, StreamWriter, open_connection, start_server
from typing import AsyncGenerator, AsyncIterable, Callable, Optional, Protocol, Type

import numpy as np
import websockets
from samplerate import Resampler

from easy_audio_interfaces.types.audio import NumpyFrame

logger = logging.getLogger(__name__)

AUDIO_FORMAT = "int16"


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


class SocketReceiver(AudioSource):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        port: int = 5000,
        host: str = "localhost",
        post_process_callback: Optional[Callable[[bytes], NumpyFrame]] = None,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._port = port
        self._host = host
        self.websocket = None
        self._server = None
        self.post_process_callback = post_process_callback

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def handle_client(self, websocket, path):
        self.websocket = websocket
        logger.info(f"Accepted connection from {websocket.remote_address}")

    async def open(self):
        self._server = await websockets.serve(self.handle_client, self._host, self._port)
        logger.info(f"WebSocket server listening on ws://{self._host}:{self._port}")
        # Wait until a client connects
        while self.websocket is None:
            await asyncio.sleep(0.1)

    async def close(self):
        if self.websocket:
            await self.websocket.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Closed WebSocket receiver.")

    async def read(self) -> NumpyFrame:
        assert self.websocket is not None, "WebSocket is not connected."
        try:
            data = await self.websocket.recv()
            if data:
                if self.post_process_callback:
                    post_process_data = self.post_process_callback(data)
                    return post_process_data
                else:
                    return NumpyFrame.frombuffer(data)
            else:
                logger.info("No data received.")
                return NumpyFrame(np.array([], dtype=np.int16))
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed.")
            return NumpyFrame(np.array([], dtype=np.int16))

    async def iter_frames(self) -> AsyncGenerator[NumpyFrame, None]:
        logger.debug("Starting frame iteration.")
        while True:
            logger.debug("Reading frame.")
            frame = await self.read()
            logger.debug(f"Received frame with {len(frame)} samples")
            if frame.size > 0:
                yield frame
            else:
                break
        logger.debug("Ending frame iteration.")


class SocketStreamer:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        port: int = 5000,
        host: str = "localhost",
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._port = port
        self._host = host
        self.writer: Optional[StreamWriter] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        reader, self.writer = await open_connection(self._host, self._port)
        logger.info(f"Connected to {self._host}:{self._port}")

    async def close(self):
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
            logger.info("Closed socket streamer.")

    async def write(self, data: NumpyFrame):
        assert self.writer is not None, "Socket is not connected."
        self.writer.write(data.tobytes())
        await self.writer.drain()

    async def write_from(self, input_stream: AsyncIterable[NumpyFrame]):
        async for chunk in input_stream:
            await self.write(chunk)


class CollectorBlock:
    def __init__(
        self,
        sample_rate: int,
        collect_seconds: float,
    ):
        self._sample_rate = sample_rate
        self._collect_samples = int(collect_seconds * sample_rate)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    async def collect(
        self, input_stream: AsyncIterable[NumpyFrame]
    ) -> AsyncGenerator[NumpyFrame, None]:
        samples = []

        async for frame in input_stream:
            samples.append(frame)
            total_samples = sum(len(s) for s in samples)
            if total_samples >= self._collect_samples:
                yield NumpyFrame(np.concatenate(samples))
                samples = []
        if samples:
            yield NumpyFrame(np.concatenate(samples))


class ResamplingBlock:
    def __init__(
        self,
        original_sample_rate: int,
        resample_rate: int,
        conversion_method: str = "sinc_fastest",
    ):
        self._original_sample_rate = original_sample_rate
        self._resample_rate = resample_rate
        self._conversion_method = conversion_method

    @property
    def sample_rate(self) -> int:
        return self._resample_rate

    async def resample(
        self, input_stream: AsyncIterable[NumpyFrame]
    ) -> AsyncGenerator[NumpyFrame, None]:
        resampler = Resampler(
            converter_type=self._conversion_method,
            channels=1,
        )
        async for frame in input_stream:
            resampled_data = resampler.process(
                frame.astype(np.float32),
                ratio=self._resample_rate / self._original_sample_rate,
            )
            yield NumpyFrame(resampled_data.astype(np.int16))


__all__ = [
    "AudioSource",
    "SocketReceiver",
    "SocketStreamer",
    "CollectorBlock",
    "ResamplingBlock",
]
