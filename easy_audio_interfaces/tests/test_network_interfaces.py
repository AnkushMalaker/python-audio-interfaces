import asyncio
from typing import AsyncGenerator, cast

import pytest
from pydub import AudioSegment, generators

from easy_audio_interfaces.network.network_interfaces import (
    SocketReceiver,
    SocketStreamer,
)

SINE_FREQUENCY = 440
SINE_SAMPLE_RATE = 44100
TEST_PORT = 8765  # Using a different port for testing


def create_sine_wave_segment(duration_ms: int) -> AudioSegment:
    sine_wave = generators.Sine(freq=SINE_FREQUENCY, sample_rate=int(SINE_SAMPLE_RATE))
    return sine_wave.to_audio_segment(duration=duration_ms)


async def async_generator(wav_segment: AudioSegment) -> AsyncGenerator[AudioSegment, None]:
    # Split the wav_segment into 1-second chunks
    chunk_duration_ms = 1000
    for i in range(0, len(wav_segment), chunk_duration_ms):
        chunk = cast(AudioSegment, wav_segment[i : i + chunk_duration_ms])
        await asyncio.sleep(0.0)  # simulate async
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
    """Test basic connection between SocketReceiver and SocketStreamer"""
    duration_ms = 3000  # 3 seconds
    wav_segment = create_sine_wave_segment(duration_ms)
    chunk_ms = 1000

    async with SocketReceiver(port=TEST_PORT) as receiver:
        async with SocketStreamer(port=TEST_PORT) as streamer:
            # Allow time for connection
            await asyncio.sleep(0.1)

            # Send a single chunk
            test_chunk = cast(AudioSegment, wav_segment[:chunk_ms])
            await streamer.write(test_chunk)

            # Receive the chunk
            received_chunk = await receiver.read()

            # Validation
            assert isinstance(received_chunk, AudioSegment)
            assert abs(len(received_chunk) - chunk_ms) < 50  # Allow small discrepancy
            assert received_chunk.frame_rate == SINE_SAMPLE_RATE


@pytest.mark.asyncio
async def test_socket_streaming():
    """Test continuous streaming of audio data"""
    duration_ms = 5000  # 5 seconds
    wav_segment = create_sine_wave_segment(duration_ms)
    expected_chunks = 5

    async with SocketReceiver(port=TEST_PORT) as receiver:
        async with SocketStreamer(port=TEST_PORT) as streamer:
            # Allow time for connection
            await asyncio.sleep(0.1)

            # Start receiving task
            receive_task = asyncio.create_task(
                async_generator_to_list(receiver.iter_frames(), max_chunks=expected_chunks)
            )

            # Stream the audio
            await streamer.write_from(async_generator(wav_segment))

            # Get received chunks
            received_chunks = await receive_task

            # Validation
            assert len(received_chunks) == expected_chunks
            for chunk in received_chunks:
                assert isinstance(chunk, AudioSegment)
                assert abs(len(chunk) - 1000) < 50  # Each chunk should be ~1 second
                assert chunk.frame_rate == SINE_SAMPLE_RATE


@pytest.mark.asyncio
async def test_socket_receiver_multiple_connections():
    """Test that SocketReceiver handles multiple connection attempts correctly"""
    async with SocketReceiver(port=TEST_PORT) as receiver:
        # Create first connection
        async with SocketStreamer(port=TEST_PORT) as streamer1:
            await asyncio.sleep(0.1)

            # Try to create second connection
            async with SocketStreamer(port=TEST_PORT) as streamer2:
                await asyncio.sleep(0.1)

                # Basic validation
                assert receiver.websocket is not None
                assert receiver.websocket.open is True


@pytest.mark.asyncio
async def test_socket_heartbeat():
    """Test that heartbeat messages are properly exchanged"""
    async with SocketReceiver(port=TEST_PORT) as receiver:
        async with SocketStreamer(port=TEST_PORT) as streamer:
            # Allow time for connection and heartbeat
            await asyncio.sleep(6)  # Wait for at least one heartbeat cycle (5s)

            # Validation
            assert receiver.websocket is not None
            assert receiver.websocket.open is True


@pytest.mark.asyncio
async def test_socket_error_handling():
    """Test error handling in SocketReceiver and SocketStreamer"""
    invalid_port = 123456  # Invalid port number

    # Test invalid receiver port
    with pytest.raises(OSError):
        async with SocketReceiver(port=invalid_port) as receiver:
            pass

    # Test connection to non-existent receiver
    with pytest.raises(ConnectionRefusedError):
        async with SocketStreamer(port=TEST_PORT) as streamer:
            pass
