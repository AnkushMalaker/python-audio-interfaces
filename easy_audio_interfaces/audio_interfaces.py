import asyncio
import logging
import wave
from pathlib import Path
from typing import AsyncGenerator, AsyncIterable, Callable, Optional, Protocol, Type

import numpy as np
import websockets
from samplerate import Resampler

from easy_audio_interfaces.types.audio import NumpyFrame
from easy_audio_interfaces.types.common import PathLike

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


class SocketReceiver(AudioSource):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        port: int = 8080,
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
        self._frame_queue: asyncio.Queue[NumpyFrame] = asyncio.Queue(
            maxsize=1000
        )  # Adjust maxsize as needed
        self._stop_event = asyncio.Event()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def handle_client(self, websocket):
        self.websocket = websocket
        logger.info(f"Accepted connection from {websocket.remote_address}")
        try:
            async for frame in self._handle_messages(websocket):
                await self._frame_queue.put(frame)
        finally:
            self.websocket = None

    async def _handle_messages(self, websocket):
        while True:
            try:
                message = await websocket.recv()
                logger.debug(f"Received {len(message)} bytes from {websocket.remote_address}")
                if self.post_process_callback:
                    yield self.post_process_callback(message)
                else:
                    yield NumpyFrame.frombuffer(message)
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected. Waiting for new connection.")
                break

    async def open(self):
        logger.debug(f"Starting WebSocket server on {self._host}:{self._port}")
        self._server = await websockets.serve(self.handle_client, self._host, self._port)
        logger.info(f"WebSocket server listening on ws://{self._host}:{self._port}")
        logger.debug("Waiting for a client connection...")
        while self.websocket is None:
            await asyncio.sleep(0.1)
        logger.debug("Client connected")

    async def read(self) -> NumpyFrame:
        if self._frame_queue.empty():
            logger.debug("Frame queue is empty, waiting for new frames...")
            await asyncio.sleep(0.1)  # Avoid busy waiting
            return NumpyFrame(np.array([], dtype=np.int16))
        frame = await self._frame_queue.get()
        logger.debug(f"Read frame of size: {len(frame)}")
        return frame

    async def iter_frames(self) -> AsyncGenerator[NumpyFrame, None]:
        while not self._stop_event.is_set():
            frame = await self.read()
            if frame.size > 0:
                yield frame
            else:
                await asyncio.sleep(0.1)  # Avoid busy waiting

    async def stop(self):
        self._stop_event.set()

    async def close(self):
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        logger.info("Closed WebSocket receiver.")

    async def __aenter__(self) -> "SocketReceiver":
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
        logger.debug("Starting frame iteration")
        while not self._stop_event.is_set():
            frame = await self.read()
            if frame.size == 0:
                logger.debug("Received empty frame, continuing...")
                continue
            logger.debug(f"Yielding frame of size: {len(frame)}")
            yield frame
        logger.debug("Frame iteration finished")


class SocketStreamer(AudioSink):
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
        self.websocket = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        uri = f"ws://{self._host}:{self._port}"
        self.websocket = await websockets.connect(uri)
        logger.info(f"Connected to {uri}")

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            logger.info("Closed WebSocket streamer.")

    async def write(self, data: NumpyFrame):
        assert self.websocket is not None, "WebSocket is not connected."
        await self.websocket.send(data.tobytes())
        logger.debug(f"Sent {len(data)} bytes to {self.websocket.remote_address}")

    async def write_from(self, input_stream: AsyncIterable[NumpyFrame]):
        async for chunk in input_stream:
            await self.write(chunk)

    async def __aiter__(self):
        yield


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


class RechunkingBlock:
    def __init__(self, chunk_size: int):
        self._chunk_size = chunk_size
        self._buffer = np.array([], dtype=np.int16)

    async def rechunk(
        self, input_stream: AsyncIterable[NumpyFrame]
    ) -> AsyncGenerator[NumpyFrame, None]:
        async for frame in input_stream:
            self._buffer = np.concatenate([self._buffer, frame])

            while len(self._buffer) >= self._chunk_size:
                chunk = self._buffer[: self._chunk_size]
                self._buffer = self._buffer[self._chunk_size :]
                yield NumpyFrame(chunk)

        # Yield any remaining samples in the buffer
        if len(self._buffer) > 0:
            yield NumpyFrame(self._buffer)
            self._buffer = np.array([], dtype=np.int16)


class LocalFileStreamer(AudioSource):
    def __init__(
        self,
        file_path: PathLike,
        chunk_size: int = 1024,
    ):
        self._file_path = Path(file_path)
        self._chunk_size = chunk_size
        self._wave_file: Optional[wave.Wave_read] = None
        self._sample_rate: int = 0
        self._channels: int = 0

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        self._wave_file = wave.open(str(self._file_path), "rb")
        self._sample_rate = self._wave_file.getframerate()
        self._channels = self._wave_file.getnchannels()
        logger.info(f"Opened file: {self._file_path}")
        logger.info(f"Sample rate: {self._sample_rate}, Channels: {self._channels}")

    async def read(self) -> NumpyFrame:
        if self._wave_file is None:
            raise RuntimeError("File is not open. Call 'open()' first.")

        chunk = self._wave_file.readframes(self._chunk_size)
        if not chunk:
            return NumpyFrame(np.array([], dtype=np.int16))

        return NumpyFrame(np.frombuffer(chunk, dtype=np.int16))

    async def close(self):
        if self._wave_file:
            self._wave_file.close()
            self._wave_file = None
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

    async def iter_frames(self) -> AsyncGenerator[NumpyFrame, None]:
        while True:
            frame = await self.read()
            if frame.size == 0:
                break
            yield frame


class LocalFileSink(AudioSink):
    def __init__(
        self,
        file_path: PathLike,
        sample_rate: int,
        channels: int,
        sample_width: int = 2,  # Default to 16-bit audio
    ):
        self._file_path = Path(file_path)
        self._sample_rate = sample_rate
        self._channels = channels
        self._sample_width = sample_width
        self._wave_file: Optional[wave.Wave_write] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        logger.debug(f"Opening file for writing: {self._file_path}")
        self._wave_file = wave.open(str(self._file_path), "wb")
        self._wave_file.setnchannels(self._channels)
        self._wave_file.setsampwidth(self._sample_width)
        self._wave_file.setframerate(self._sample_rate)
        logger.info(f"Opened file for writing: {self._file_path}")

    async def write(self, data: NumpyFrame):
        if self._wave_file is None:
            raise RuntimeError("File is not open. Call 'open()' first.")
        self._wave_file.writeframes(data.tobytes())
        logger.debug(f"Wrote {len(data)} bytes to {self._file_path}")

    async def write_from(self, input_stream: AsyncIterable[NumpyFrame]):
        total_frames = 0
        total_bytes = 0
        async for chunk in input_stream:
            await self.write(chunk)
            total_frames += 1
            total_bytes += len(chunk)
        logger.info(
            f"Finished writing {total_frames} frames ({total_bytes} bytes) to {self._file_path}"
        )

    async def close(self):
        if self._wave_file:
            self._wave_file.close()
            self._wave_file = None
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


__all__ = [
    "AudioSource",
    "AudioSink",
    "SocketReceiver",
    "SocketStreamer",
    "CollectorBlock",
    "ResamplingBlock",
    "RechunkingBlock",
    "LocalFileStreamer",
    "LocalFileSink",
]
