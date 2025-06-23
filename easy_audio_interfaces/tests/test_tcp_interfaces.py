import asyncio

import pytest
from wyoming.audio import AudioChunk

from easy_audio_interfaces.network.network_interfaces import TCPClient, TCPServer

from .utils import create_sine_wave_audio_chunk

SINE_FREQUENCY = 440
SINE_SAMPLE_RATE = 44100
BASE_TEST_PORT = 9100  # Different base port to avoid conflicts with WebSocket tests


@pytest.mark.asyncio
async def test_tcp_server_properties():
    """Test TCPServer properties"""
    sample_rate = 22050
    channels = 2

    server = TCPServer(sample_rate=sample_rate, channels=channels)

    assert server.sample_rate == sample_rate
    assert server.channels == channels


@pytest.mark.asyncio
async def test_tcp_client_properties():
    """Test TCPClient properties"""
    sample_rate = 22050
    channels = 2

    client = TCPClient(sample_rate=sample_rate, channels=channels)

    assert client.sample_rate == sample_rate
    assert client.channels == channels


@pytest.mark.asyncio
async def test_tcp_basic_connection():
    """Test basic TCP connection and data transfer"""
    test_port = BASE_TEST_PORT + 1

    server = TCPServer(port=test_port, sample_rate=SINE_SAMPLE_RATE, channels=1)
    client = TCPClient(port=test_port, sample_rate=SINE_SAMPLE_RATE, channels=1)

    try:
        # Start server
        server_task = asyncio.create_task(server.open())
        await asyncio.sleep(0.1)  # Allow server to start

        # Connect client with timeout
        await asyncio.wait_for(client.open(), timeout=2.0)
        await asyncio.wait_for(server_task, timeout=2.0)

        # Create and send small audio chunk
        audio_chunk = create_sine_wave_audio_chunk(100, SINE_FREQUENCY, SINE_SAMPLE_RATE)
        await asyncio.wait_for(client.write(audio_chunk), timeout=2.0)

        # Receive the chunk
        received_chunk = await asyncio.wait_for(server.read(), timeout=2.0)

        # Validation
        assert isinstance(received_chunk, AudioChunk)
        assert received_chunk.rate == SINE_SAMPLE_RATE
        assert len(received_chunk.audio) > 0

    except asyncio.TimeoutError:
        pytest.fail("Test timed out")
    finally:
        # Cleanup with timeouts
        try:
            await asyncio.wait_for(client.close(), timeout=1.0)
        except:
            pass
        try:
            await asyncio.wait_for(server.close(), timeout=1.0)
        except:
            pass


@pytest.mark.asyncio
async def test_tcp_error_handling():
    """Test error handling in TCPServer and TCPClient"""

    # Test binding to invalid port (use port 0 which should work, but 65536 which should fail)
    with pytest.raises((OSError, OverflowError)):
        server = TCPServer(port=65536)  # Port out of range
        await asyncio.wait_for(server.open(), timeout=1.0)

    # Test connecting to non-existent server
    with pytest.raises((ConnectionRefusedError, OSError, asyncio.TimeoutError)):
        client = TCPClient(port=BASE_TEST_PORT + 99)
        await asyncio.wait_for(client.open(), timeout=1.0)


@pytest.mark.asyncio
async def test_tcp_write_without_connection():
    """Test writing to TCPClient without connection"""
    client = TCPClient()
    chunk = create_sine_wave_audio_chunk(100, SINE_FREQUENCY, SINE_SAMPLE_RATE)

    with pytest.raises(RuntimeError):
        await client.write(chunk)


@pytest.mark.asyncio
async def test_tcp_read_without_connection():
    """Test reading from TCPServer without connection"""
    server = TCPServer()

    result = await server.read()
    assert result is None


@pytest.mark.asyncio
async def test_tcp_custom_post_process():
    """Test TCPServer with custom post-processing function"""
    test_port = BASE_TEST_PORT + 2

    def custom_processor(data: bytes) -> AudioChunk:
        return AudioChunk(
            audio=data,
            width=2,
            rate=48000,  # Different sample rate
            channels=2,  # Different channels
        )

    server = TCPServer(port=test_port, post_process_bytes_fn=custom_processor)
    client = TCPClient(port=test_port)

    try:
        # Start server and connect client
        server_task = asyncio.create_task(server.open())
        await asyncio.sleep(0.1)
        await asyncio.wait_for(client.open(), timeout=2.0)
        await asyncio.wait_for(server_task, timeout=2.0)

        # Send data
        chunk = create_sine_wave_audio_chunk(100, SINE_FREQUENCY, SINE_SAMPLE_RATE)
        await asyncio.wait_for(client.write(chunk), timeout=2.0)

        # Receive with custom processing
        received = await asyncio.wait_for(server.read(), timeout=2.0)
        assert received is not None
        assert received.rate == 48000  # Custom sample rate
        assert received.channels == 2  # Custom channels

    except asyncio.TimeoutError:
        pytest.fail("Test timed out")
    finally:
        try:
            await asyncio.wait_for(client.close(), timeout=1.0)
        except:
            pass
        try:
            await asyncio.wait_for(server.close(), timeout=1.0)
        except:
            pass


@pytest.mark.asyncio
async def test_tcp_client_disconnection():
    """Test handling of client disconnection"""
    test_port = BASE_TEST_PORT + 3

    server = TCPServer(port=test_port, sample_rate=SINE_SAMPLE_RATE, channels=1)
    client = TCPClient(port=test_port, sample_rate=SINE_SAMPLE_RATE, channels=1)

    try:
        # Start server and connect client
        server_task = asyncio.create_task(server.open())
        await asyncio.sleep(0.1)
        await asyncio.wait_for(client.open(), timeout=2.0)
        await asyncio.wait_for(server_task, timeout=2.0)

        # Verify connection is working
        chunk = create_sine_wave_audio_chunk(100, SINE_FREQUENCY, SINE_SAMPLE_RATE)
        await asyncio.wait_for(client.write(chunk), timeout=2.0)

        received = await asyncio.wait_for(server.read(), timeout=2.0)
        assert received is not None
        assert len(received.audio) > 0

        # Close client
        await asyncio.wait_for(client.close(), timeout=1.0)

        # Wait longer for socket closure to be detected
        await asyncio.sleep(0.5)

        # Server should eventually detect disconnection
        # May receive remaining buffered data first, then None
        disconnect_detected = False
        for _ in range(3):  # Try up to 3 reads
            try:
                received = await asyncio.wait_for(server.read(), timeout=1.0)
                if received is None:
                    disconnect_detected = True
                    break
            except asyncio.TimeoutError:
                disconnect_detected = True  # Timeout also indicates disconnection
                break

        assert disconnect_detected, "Server should detect client disconnection"

    except asyncio.TimeoutError:
        pytest.fail("Test timed out")
    finally:
        try:
            await asyncio.wait_for(server.close(), timeout=1.0)
        except:
            pass
