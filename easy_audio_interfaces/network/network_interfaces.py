import asyncio
import logging
import time
from asyncio import Task
from collections.abc import Coroutine
from typing import Any, AsyncGenerator, AsyncIterable, Callable, Optional, Type

import websockets
from pydub import AudioSegment

from easy_audio_interfaces.base_interfaces import AudioSink, AudioSource

logger = logging.getLogger(__name__)


class SocketReceiver(AudioSource):
    """
    A class that represents a WebSocket audio source receiver.

    This class allows for receiving audio data over a WebSocket connection. It handles
    client connections, processes incoming audio frames, and manages the WebSocket server.

    Attributes:
        sample_rate (int): The sample rate of the audio (default is 16000 Hz).
        channels (int): The number of audio channels (default is 1).
        port (int): The port on which the WebSocket server listens (default is 8080).
        host (str): The host address for the WebSocket server (default is "localhost").
        post_process_bytes_fn (Optional[Callable[[bytes], AudioSegment]]): A function to process
            incoming byte data into a AudioSegment.
        server_routine (Optional[Coroutine[Any, Any, None]]): An optional coroutine that runs
            the server routine, defaults to a heartbeat function.

    Methods:
        handle_client(websocket): Handles incoming client connections and messages.
        open(): Starts the WebSocket server and waits for a client connection.
        read() -> AudioSegment: Reads a frame from the frame queue.
        iter_frames() -> AsyncGenerator[AudioSegment, None]: Asynchronously iterates over received frames.
        stop(): Signals to stop the receiver.
        close(): Closes the WebSocket server and cleans up resources.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        port: int = 8080,
        host: str = "localhost",
        post_process_bytes_fn: Optional[Callable[[bytes], AudioSegment]] = None,
        server_routine: Optional[Coroutine[Any, Any, None]] = None,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._port = port
        self._host = host
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._server = None
        self.post_process_bytes_fn = post_process_bytes_fn
        self._frame_queue: asyncio.Queue[AudioSegment] = asyncio.Queue(
            maxsize=1000
        )  # Adjust maxsize as needed
        self._stop_event = asyncio.Event()
        self._server_routine = server_routine or self._send_heartbeat()
        self._server_task: Optional[Task[Any | None]] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol):
        if self.websocket:
            logger.warning(
                "Should only have one client per socket receiver. Check for logical error. Closing existing connection."
            )
            await self.websocket.close()
        self.websocket = websocket

        logger.info(f"Accepted connection from {websocket.remote_address}")

        self._server_task = asyncio.create_task(self._server_routine)

        await websocket.send("ack")

        frame_counter = 0
        try:
            async for frame in self._handle_messages(websocket):
                frame_counter += 1
                await self._frame_queue.put(frame)
        finally:
            self.websocket = None
            if self._server_task:
                self._server_task.cancel()
            await self.stop()

    async def _send_heartbeat(self):
        while self.websocket and self.websocket.open:
            try:
                await self.websocket.send("heartbeat")
                logger.debug("Heartbeat sent")
            except websockets.exceptions.WebSocketException:
                logger.warning("Failed to send heartbeat")
            await asyncio.sleep(5)  # Send heartbeat every 5 seconds

    async def _handle_messages(self, websocket: websockets.WebSocketServerProtocol):
        while self.websocket and self.websocket.open:
            try:
                message = await websocket.recv()
                if message == "heartbeat":
                    logger.debug("Heartbeat received")
                    self._last_recv_heartbeat = time.time()
                    continue
                logger.debug(f"Received {len(message)} bytes from {websocket.remote_address}")
                if self.post_process_bytes_fn:
                    yield self.post_process_bytes_fn(message)  # type: ignore
                else:
                    yield AudioSegment.from_raw_data(message)  # type: ignore
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected. Waiting for new connection.")
                break

    async def open(self):
        logger.debug(f"Starting WebSocket server on {self._host}:{self._port}")
        self._server = await websockets.serve(self.handle_client, self._host, self._port)
        logger.info(f"WebSocket server listening on ws://{self._host}:{self._port}")
        logger.debug("Waiting for a client connection...")
        while self.websocket is None:
            logger.debug("Waiting for a client connection...")
            await asyncio.sleep(0.1)
        logger.debug("Client connected")

    async def read(self) -> AudioSegment:
        frame = await self._frame_queue.get()
        logger.debug(f"Read frame of size: {len(frame)}")
        return frame

    async def iter_frames(self) -> AsyncGenerator[AudioSegment, None]:
        while not self._stop_event.is_set():
            yield await self.read()

    async def stop(self):
        self._stop_event.set()

    async def close(self):
        await self.stop()
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
        async for frame in self.iter_frames():
            yield frame


class SocketStreamer(AudioSink):
    """
    A class that represents a WebSocket audio sink streamer.

    This class allows for sending audio data over a WebSocket connection. It handles
    client connections, processes incoming audio frames, and manages the WebSocket server.

    Attributes:
        sample_rate (int): The sample rate of the audio (default is 16000 Hz).
        channels (int): The number of audio channels (default is 1).
        port (int): The port on which the WebSocket server listens (default is 8080).
        host (str): The host address for the WebSocket server (default is "localhost").

    Methods:
        open(): Connects to the WebSocket server and waits for a client connection.
        write(data: NumpyFrame): Sends a frame of audio data to the WebSocket server.
        write_from(input_stream: AsyncIterable[NumpyFrame]): Writes audio data from an input stream to the WebSocket server.
        close(): Closes the WebSocket connection and cleans up resources.

    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        port: int = 8080,
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

    async def write(self, data: AudioSegment):
        assert self.websocket is not None, "WebSocket is not connected."
        # Convert AudioSegment to bytes
        raw_data = data.raw_data
        await self.websocket.send(raw_data)  # type: ignore
        logger.debug(f"Sent {len(raw_data)} bytes to {self.websocket.remote_address}")  # type: ignore

    async def write_from(self, input_stream: AsyncIterable[AudioSegment]):
        async for chunk in input_stream:
            await self.write(chunk)

    async def __aiter__(self):
        yield
