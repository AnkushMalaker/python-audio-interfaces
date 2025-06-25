import asyncio
import logging
import socket
import time
from asyncio import Task
from collections.abc import Coroutine
from typing import Any, AsyncGenerator, AsyncIterable, Callable, Optional, Type

import websockets
from wyoming.audio import AudioChunk

from easy_audio_interfaces.base_interfaces import AudioSink, AudioSource

logger = logging.getLogger(__name__)


class SocketServer(AudioSource):
    """
    A class that represents a WebSocket audio source receiver.

    This class allows for receiving audio data over a WebSocket connection. It handles
    client connections, processes incoming audio frames, and manages the WebSocket server.

    Attributes:
        sample_rate (int): The sample rate of the audio (default is 16000 Hz).
        channels (int): The number of audio channels (default is 1).
        port (int): The port on which the WebSocket server listens (default is 8080).
        host (str): The host address for the WebSocket server (default is "localhost").
        post_process_bytes_fn (Optional[Callable[[bytes], AudioChunk]]): A function to process
            incoming byte data into a AudioChunk.
        server_routine (Optional[Coroutine[Any, Any, None]]): An optional coroutine that runs
            the server routine, defaults to a heartbeat function.

    Methods:
        handle_client(websocket): Handles incoming client connections and messages.
        open(): Starts the WebSocket server and waits for a client connection.
        read() -> AudioChunk: Reads a frame from the frame queue.
        iter_frames() -> AsyncGenerator[AudioChunk, None]: Asynchronously iterates over received frames.
        stop(): Signals to stop the receiver.
        close(): Closes the WebSocket server and cleans up resources.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        port: int = 8080,
        host: str = "localhost",
        post_process_bytes_fn: Optional[Callable[[bytes], AudioChunk]] = None,
        server_routine: Optional[Coroutine[Any, Any, None]] = None,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._port = port
        self._host = host
        self.websocket: Optional[websockets.ServerConnection] = None
        self._server = None
        self.post_process_bytes_fn = post_process_bytes_fn
        self._frame_queue: asyncio.Queue[AudioChunk] = asyncio.Queue(
            maxsize=1000
        )  # Adjust maxsize as needed
        self._stop_event = asyncio.Event()
        self._server_routine = server_routine
        self._server_task: Optional[Task[Any | None]] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def handle_client(self, websocket: websockets.ServerConnection):
        if self.websocket:
            logger.warning(
                "Should only have one client per socket receiver. Check for logical error. Closing existing connection."
            )
            await self.websocket.close()
        self.websocket = websocket

        logger.info(f"Accepted connection from {websocket.remote_address}")

        if self._server_routine:
            self._server_task = asyncio.create_task(self._server_routine)
        else:
            self._server_task = asyncio.create_task(self._send_heartbeat())

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
        while self.websocket and self.websocket.server.is_serving():
            try:
                await self.websocket.send("heartbeat")
                logger.debug("Heartbeat sent")
            except websockets.exceptions.WebSocketException:
                logger.warning("Failed to send heartbeat")
            await asyncio.sleep(5)  # Send heartbeat every 5 seconds

    async def _handle_messages(self, websocket: websockets.ServerConnection):
        while self.websocket and self.websocket.server.is_serving():
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
                    # Convert message to bytes if needed
                    if isinstance(message, bytes):
                        audio_bytes = message
                    elif isinstance(message, str):
                        audio_bytes = message.encode()
                    else:
                        audio_bytes = bytes(message)

                    yield AudioChunk(
                        audio=audio_bytes,
                        width=2,
                        rate=self._sample_rate,
                        channels=self._channels,
                    )
            except websockets.exceptions.ConnectionClosed:
                logger.info("Client disconnected. Waiting for new connection.")
                break

    async def open(self):
        logger.debug(f"Starting WebSocket server on {self._host}:{self._port}")
        self._server = await websockets.serve(self.handle_client, self._host, self._port)
        logger.info(f"WebSocket server listening on ws://{self._host}:{self._port}")
        # Server is ready, don't wait for client connection here

    async def read(self) -> AudioChunk:
        frame = await self._frame_queue.get()
        logger.debug(f"Read frame of size: {frame.samples}")
        return frame

    async def iter_frames(self) -> AsyncGenerator[AudioChunk, None]:
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

    async def __aenter__(self) -> "SocketServer":
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


class SocketClient(AudioSink):
    """
    A class that represents a WebSocket audio sink streamer.

    Attributes:
        sample_rate (int): The sample rate of the audio (default is 16000 Hz).
        channels (int): The number of audio channels (default is 1).
        uri (str): The WebSocket URI to connect to (default is "ws://localhost:8080").

    Methods:
        open(): Connects to the WebSocket server and waits for a client connection.
        write(data: AudioChunk): Sends a frame of audio data to the WebSocket server.
        write_from(input_stream: AsyncIterable[AudioChunk]): Writes audio data from an input stream to the WebSocket server.
        close(): Closes the WebSocket connection and cleans up resources.

    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        uri: str = "ws://localhost:8080",
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._uri = uri
        self.websocket = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        self.websocket = await websockets.connect(self._uri)
        logger.info(f"Connected to {self._uri}")

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            logger.info("Closed WebSocket streamer.")

    async def write(self, data: AudioChunk):
        assert self.websocket is not None, "WebSocket is not connected."
        # Convert AudioSegment to bytes
        raw_data = data.audio
        await self.websocket.send(raw_data)  # type: ignore
        logger.debug(f"Sent {len(raw_data)} bytes to {self.websocket.remote_address}")  # type: ignore

    async def write_from(self, input_stream: AsyncIterable[AudioChunk]):
        async for chunk in input_stream:
            await self.write(chunk)

    async def __aiter__(self):
        yield


class TCPServer(AudioSource):
    """
    A class that represents a TCP audio source receiver.

    This class allows for receiving audio data over a TCP connection. It handles
    client connections, processes incoming audio data, and manages the TCP server.

    Attributes:
        sample_rate (int): The sample rate of the audio (default is 16000 Hz).
        channels (int): The number of audio channels (default is 2).
        port (int): The port on which the TCP server listens (default is 8989).
        host (str): The host address for the TCP server (default is "0.0.0.0").
        sample_width (int): The sample width in bytes (default is 2 for 16-bit).
        chunk_size (int): The size of data chunks to read from the socket (default is 4096).
        post_process_bytes_fn (Optional[Callable[[bytes], AudioChunk]]): A function to process
            incoming byte data into an AudioChunk.

    Methods:
        open(): Starts the TCP server and waits for a client connection.
        read() -> Optional[AudioChunk]: Reads audio data from the TCP client.
        iter_frames() -> AsyncGenerator[AudioChunk, None]: Asynchronously iterates over received frames.
        close(): Closes the TCP server and client connections.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        port: int = 8989,
        host: str = "0.0.0.0",
        sample_width: int = 2,
        chunk_size: int = 4096,
        post_process_bytes_fn: Optional[Callable[[bytes], AudioChunk]] = None,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._port = port
        self._host = host
        self._sample_width = sample_width
        self._chunk_size = chunk_size
        self.post_process_bytes_fn = post_process_bytes_fn

        self._server_socket: Optional[socket.socket] = None
        self._client_socket: Optional[socket.socket] = None
        self._client_addr: Optional[tuple] = None
        self._is_running = False

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        """Start the TCP server and wait for a client connection."""
        self._server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server_socket.bind((self._host, self._port))
        self._server_socket.listen(1)
        self._server_socket.setblocking(False)

        logger.info(f"TCP server listening on {self._host}:{self._port}")

        # Wait for client connection
        loop = asyncio.get_event_loop()
        self._client_socket, self._client_addr = await loop.sock_accept(self._server_socket)
        self._client_socket.setblocking(False)

        if self._client_addr:
            logger.info(f"Client connected from {self._client_addr[0]}:{self._client_addr[1]}")
        self._is_running = True

    async def read(self) -> Optional[AudioChunk]:
        """Read audio data from the TCP client."""
        if not self._is_running or not self._client_socket:
            return None

        try:
            loop = asyncio.get_event_loop()
            data = await loop.sock_recv(self._client_socket, self._chunk_size)

            if not data:
                logger.warning("Client disconnected")
                self._is_running = False
                return None

            # Use post-processing function if provided, otherwise create default AudioChunk
            if self.post_process_bytes_fn:
                return self.post_process_bytes_fn(data)
            else:
                return AudioChunk(
                    audio=data,
                    width=self._sample_width,
                    rate=self._sample_rate,
                    channels=self._channels,
                )

        except ConnectionResetError:
            logger.info("Client disconnected")
            self._is_running = False
            return None
        except Exception as e:
            logger.error(f"Error reading from client: {e}")
            self._is_running = False
            return None

    async def close(self):
        """Close the TCP server and client connections."""
        self._is_running = False

        if self._client_socket:
            self._client_socket.close()
            self._client_socket = None

        if self._server_socket:
            self._server_socket.close()
            self._server_socket = None

        logger.info("TCP server closed")

    async def iter_frames(self) -> AsyncGenerator[AudioChunk, None]:
        """Iterate over audio frames from the TCP client."""
        if not self._is_running:
            raise RuntimeError("TCP server is not running. Call 'open()' first.")
        while self._is_running:
            chunk = await self.read()
            if chunk is None:
                break
            yield chunk

    async def __aenter__(self) -> "TCPServer":
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


class TCPClient(AudioSink):
    """
    A class that represents a TCP audio sink streamer.

    This class allows for sending audio data over a TCP connection. It handles
    connecting to a TCP server and sending audio frames.

    Attributes:
        sample_rate (int): The sample rate of the audio (default is 16000 Hz).
        channels (int): The number of audio channels (default is 2).
        port (int): The port to connect to (default is 8989).
        host (str): The host address to connect to (default is "localhost").
        sample_width (int): The sample width in bytes (default is 2 for 16-bit).

    Methods:
        open(): Connects to the TCP server.
        write(data: AudioChunk): Sends audio data to the TCP server.
        write_from(input_stream: AsyncIterable[AudioChunk]): Writes audio data from an input stream to the TCP server.
        close(): Closes the TCP connection.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 2,
        port: int = 8989,
        host: str = "localhost",
        sample_width: int = 2,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._port = port
        self._host = host
        self._sample_width = sample_width
        self._socket: Optional[socket.socket] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        """Connect to the TCP server."""
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setblocking(False)

        loop = asyncio.get_event_loop()
        await loop.sock_connect(self._socket, (self._host, self._port))
        logger.info(f"Connected to TCP server at {self._host}:{self._port}")

    async def close(self):
        """Close the TCP connection."""
        if self._socket:
            self._socket.close()
            self._socket = None
            logger.info("Closed TCP client connection")

    async def write(self, data: AudioChunk):
        """Send audio data to the TCP server."""
        if self._socket is None:
            raise RuntimeError("Socket is not connected. Call 'open()' first.")

        loop = asyncio.get_event_loop()
        await loop.sock_sendall(self._socket, data.audio)
        logger.debug(f"Sent {len(data.audio)} bytes to TCP server")

    async def write_from(self, input_stream: AsyncIterable[AudioChunk]):
        """Write audio data from an input stream to the TCP server."""
        total_bytes = 0
        chunk_count = 0
        async for chunk in input_stream:
            await self.write(chunk)
            total_bytes += len(chunk.audio)
            chunk_count += 1
        logger.info(f"Sent {chunk_count} chunks ({total_bytes} bytes) to TCP server")

    async def __aenter__(self) -> "TCPClient":
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
        # TCP client doesn't produce data, so this is an empty async generator
        return
        yield  # This line will never be reached, but needed for typing
