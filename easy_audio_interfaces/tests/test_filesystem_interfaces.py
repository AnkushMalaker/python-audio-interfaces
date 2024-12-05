import asyncio
from pathlib import Path
from typing import AsyncGenerator, cast

import pytest
from pydub import AudioSegment, generators

from easy_audio_interfaces.filesystem.filesystem_interfaces import (
    LocalFileSink,
    LocalFileStreamer,
)

SINE_FREQUENCY = 440
SINE_SAMPLE_RATE = 44100
TEST_FILE_PATH = "test_audio.wav"


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


@pytest.mark.asyncio
async def test_local_file_sink_and_streamer():
    """Test writing to and reading from a local file"""
    duration_ms = 5000  # 5 seconds
    wav_segment = create_sine_wave_segment(duration_ms)
    chunk_ms = 20

    # Write to file using LocalFileSink
    async with LocalFileSink(TEST_FILE_PATH, sample_rate=SINE_SAMPLE_RATE, channels=1) as file_sink:
        await file_sink.write_from(async_generator(wav_segment))

    # Read from file using LocalFileStreamer
    async with LocalFileStreamer(TEST_FILE_PATH, chunk_size_ms=chunk_ms) as file_streamer:
        read_segment = await file_streamer.read()

    # Validate the read segment
    assert isinstance(read_segment, AudioSegment)
    assert abs(len(read_segment) - chunk_ms) <= 1  # Allow small discrepancy
    assert read_segment.frame_rate == SINE_SAMPLE_RATE
    assert read_segment.channels == 1

    # Clean up
    Path(TEST_FILE_PATH).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_local_file_streamer_iter_frames():
    """Test iterating over frames using LocalFileStreamer"""
    duration_ms = 5000  # 5 seconds
    wav_segment = create_sine_wave_segment(duration_ms)

    # Write to file using LocalFileSink
    async with LocalFileSink(TEST_FILE_PATH, sample_rate=SINE_SAMPLE_RATE, channels=1) as file_sink:
        await file_sink.write_from(async_generator(wav_segment))

    # Read from file using LocalFileStreamer
    async with LocalFileStreamer(TEST_FILE_PATH, chunk_size_ms=20) as file_streamer:
        frames = []
        async for frame in file_streamer.iter_frames():
            frames.append(frame)

    # Validate the frames
    assert len(frames) > 0
    for frame in frames:
        assert isinstance(frame, AudioSegment)

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
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        async with LocalFileStreamer(non_existent_file, chunk_size_ms=20):
            pass
