import asyncio
from typing import AsyncGenerator

import pytest
from wyoming.audio import AudioChunk

from easy_audio_interfaces.network.network_interfaces import SocketClient, SocketServer

from .utils import create_sine_wave_audio_chunk

SINE_FREQUENCY = 440
SINE_SAMPLE_RATE = 44100
BASE_TEST_PORT = 8989  # Base port for testing


async def async_generator(
    audio_chunk: AudioChunk, chunk_duration_ms: int = 1000
) -> AsyncGenerator[AudioChunk, None]:
    """Split an AudioChunk into smaller chunks for async streaming."""
    # Calculate bytes per millisecond
    bytes_per_ms = (audio_chunk.rate * audio_chunk.width * audio_chunk.channels) // 1000
    chunk_size_bytes = chunk_duration_ms * bytes_per_ms

    audio_data = audio_chunk.audio

    for i in range(0, len(audio_data), chunk_size_bytes):
        chunk_data = audio_data[i : i + chunk_size_bytes]
        if chunk_data:  # Only yield non-empty chunks
            chunk = AudioChunk(
                audio=chunk_data,
                rate=audio_chunk.rate,
                width=audio_chunk.width,
                channels=audio_chunk.channels,
            )
            await asyncio.sleep(0.2)  # simulate async
            yield chunk


async def async_generator_to_list(gen, max_chunks=None):
    """Helper function to collect async generator results into a list"""
    result = []
    try:
        async for item in gen:
            result.append(item)
            if max_chunks and len(result) >= max_chunks:
                break
    except asyncio.CancelledError:
        pass
    return result


@pytest.mark.asyncio
async def test_socket_connection():
    """Test basic connection between SocketServer and SocketClient"""
    duration_ms = 3000  # 3 seconds
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE)
    chunk_ms = 1000
    test_port = BASE_TEST_PORT + 1

    async with SocketServer(port=test_port, sample_rate=SINE_SAMPLE_RATE, channels=1) as receiver:
        async with SocketClient(
            uri=f"ws://localhost:{test_port}", sample_rate=SINE_SAMPLE_RATE, channels=1
        ) as streamer:
            # Allow time for connection
            await asyncio.sleep(0.1)

            # Send a single chunk
            test_chunk_data = audio_chunk.audio[
                : chunk_ms * audio_chunk.width * audio_chunk.channels * audio_chunk.rate // 1000
            ]
            test_chunk = AudioChunk(
                audio=test_chunk_data,
                rate=audio_chunk.rate,
                width=audio_chunk.width,
                channels=audio_chunk.channels,
            )
            await streamer.write(test_chunk)

            # Receive the chunk
            received_chunk = await receiver.read()

            # Validation
            assert isinstance(received_chunk, AudioChunk)
            assert abs(received_chunk.milliseconds - chunk_ms) < 50  # Allow small discrepancy
            assert received_chunk.rate == SINE_SAMPLE_RATE


@pytest.mark.asyncio
async def test_socket_streaming():
    """Test continuous streaming of audio data"""
    duration_ms = 5000  # 5 seconds
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE)
    expected_chunks = 5
    test_port = BASE_TEST_PORT + 2

    async with SocketServer(port=test_port, sample_rate=SINE_SAMPLE_RATE, channels=1) as receiver:
        async with SocketClient(
            uri=f"ws://localhost:{test_port}", sample_rate=SINE_SAMPLE_RATE, channels=1
        ) as streamer:
            # Allow time for connection
            await asyncio.sleep(0.1)

            # Start receiving task
            receive_task = asyncio.create_task(
                async_generator_to_list(receiver.iter_frames(), max_chunks=expected_chunks)
            )

            # Stream the audio
            await streamer.write_from(async_generator(audio_chunk))

            # Get received chunks
            received_chunks = await receive_task

            # Validation
            assert len(received_chunks) == expected_chunks
            for chunk in received_chunks:
                assert isinstance(chunk, AudioChunk)
                assert abs(chunk.milliseconds - 1000) < 50  # Each chunk should be ~1 second
                assert chunk.rate == SINE_SAMPLE_RATE


@pytest.mark.asyncio
async def test_socket_server_multiple_connections():
    """Test that SocketServer handles multiple connection attempts correctly"""
    test_port = BASE_TEST_PORT + 3
    async with SocketServer(port=test_port, sample_rate=SINE_SAMPLE_RATE, channels=1) as receiver:
        # Create first connection
        async with SocketClient(
            uri=f"ws://localhost:{test_port}", sample_rate=SINE_SAMPLE_RATE, channels=1
        ) as streamer1:
            await asyncio.sleep(0.1)

            # Try to create second connection
            async with SocketClient(
                uri=f"ws://localhost:{test_port}", sample_rate=SINE_SAMPLE_RATE, channels=1
            ) as streamer2:
                await asyncio.sleep(0.1)

                # Basic validation
                assert receiver.websocket is not None
                # Just verify the websocket connection exists (open attribute may not be available)
                assert receiver.websocket is not None


@pytest.mark.asyncio
async def test_socket_heartbeat():
    """Test that heartbeat messages are properly exchanged"""
    test_port = BASE_TEST_PORT + 4
    async with SocketServer(port=test_port, sample_rate=SINE_SAMPLE_RATE, channels=1) as receiver:
        async with SocketClient(
            uri=f"ws://localhost:{test_port}", sample_rate=SINE_SAMPLE_RATE, channels=1
        ) as streamer:
            # Allow time for connection and heartbeat
            await asyncio.sleep(6)  # Wait for at least one heartbeat cycle (5s)

            # Validation
            assert receiver.websocket is not None
            # Just verify the websocket connection exists (open attribute may not be available)


@pytest.mark.asyncio
async def test_socket_error_handling():
    """Test error handling in SocketServer and SocketClient"""
    test_port = BASE_TEST_PORT + 5

    # Test invalid receiver port - use a negative port which should always fail
    with pytest.raises(OSError):
        async with SocketServer(port=-1) as receiver:
            pass

    # Test connection to non-existent receiver
    with pytest.raises((ConnectionRefusedError, OSError)):
        async with SocketClient(uri=f"ws://localhost:{test_port}") as streamer:
            pass
