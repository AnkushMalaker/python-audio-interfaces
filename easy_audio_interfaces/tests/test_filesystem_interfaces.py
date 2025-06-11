from pathlib import Path

import pytest
from wyoming.audio import AudioChunk

from easy_audio_interfaces.filesystem.filesystem_interfaces import (
    LocalFileSink,
    LocalFileStreamer,
)

from .utils import async_generator, create_sine_wave_audio_chunk

SINE_FREQUENCY = 440
SINE_SAMPLE_RATE = 44100
TEST_FILE_PATH = "test_audio.wav"


@pytest.mark.asyncio
async def test_local_file_sink_and_streamer():
    """Test writing to and reading from a local file"""
    duration_ms = 5000  # 5 seconds
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE)
    chunk_ms = 20

    # Write to file using LocalFileSink
    async with LocalFileSink(TEST_FILE_PATH, sample_rate=SINE_SAMPLE_RATE, channels=1) as file_sink:
        await file_sink.write_from(async_generator(audio_chunk))

    # Read from file using LocalFileStreamer
    async with LocalFileStreamer(TEST_FILE_PATH, chunk_size_ms=chunk_ms) as file_streamer:
        read_chunk = await file_streamer.read()

    # Validate the read chunk
    assert isinstance(read_chunk, AudioChunk)
    # Check that chunk duration is approximately what we expect
    expected_samples = (chunk_ms * SINE_SAMPLE_RATE) // 1000
    assert (
        abs(read_chunk.samples - expected_samples) <= expected_samples * 0.1
    )  # Allow 10% discrepancy
    assert read_chunk.rate == SINE_SAMPLE_RATE
    assert read_chunk.channels == 1

    # Clean up
    Path(TEST_FILE_PATH).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_local_file_streamer_iter_frames():
    """Test iterating over frames using LocalFileStreamer"""
    duration_ms = 5000  # 5 seconds
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE)

    # Write to file using LocalFileSink
    async with LocalFileSink(TEST_FILE_PATH, sample_rate=SINE_SAMPLE_RATE, channels=1) as file_sink:
        await file_sink.write_from(async_generator(audio_chunk))

    # Read from file using LocalFileStreamer
    async with LocalFileStreamer(TEST_FILE_PATH, chunk_size_ms=20) as file_streamer:
        frames = []
        async for frame in file_streamer.iter_frames():
            frames.append(frame)

    # Validate the frames
    assert len(frames) > 0
    for frame in frames:
        assert isinstance(frame, AudioChunk)

    # Clean up
    Path(TEST_FILE_PATH).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_local_file_sink_error_handling():
    """Test error handling in LocalFileSink"""
    invalid_path = "/invalid/path/test_audio.wav"
    with pytest.raises(RuntimeError, match="Parent directory does not exist"):
        async with LocalFileSink(invalid_path, sample_rate=SINE_SAMPLE_RATE, channels=1):
            pass


@pytest.mark.asyncio
async def test_local_file_streamer_error_handling():
    """Test error handling in LocalFileStreamer"""
    non_existent_file = "non_existent_file.wav"
    with pytest.raises(FileNotFoundError, match="File not found:"):
        async with LocalFileStreamer(non_existent_file, chunk_size_ms=20):
            pass
