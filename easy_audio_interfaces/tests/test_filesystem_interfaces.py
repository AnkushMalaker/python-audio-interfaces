import shutil
from pathlib import Path

import pytest
from wyoming.audio import AudioChunk

from easy_audio_interfaces.filesystem.filesystem_interfaces import (
    LocalFileSink,
    LocalFileStreamer,
    RollingFileSink,
)

from .utils import async_generator, create_sine_wave_audio_chunk

SINE_FREQUENCY = 440
SINE_SAMPLE_RATE = 44100
TEST_FILE_PATH = "test_audio.wav"
TEST_ROLLING_DIR = "test_rolling_output"


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


@pytest.mark.asyncio
async def test_rolling_file_sink_basic():
    """Test basic functionality of RollingFileSink"""
    output_dir = Path(TEST_ROLLING_DIR)
    prefix = "test_recording_basic"  # Unique prefix
    segment_duration = 1.0  # 1 second segments

    # Create test audio (2 seconds, should create 2 files)
    duration_ms = 2000
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE)

    async with RollingFileSink(
        directory=output_dir,
        prefix=prefix,
        segment_duration_seconds=segment_duration,
        sample_rate=SINE_SAMPLE_RATE,
        channels=1,
    ) as rolling_sink:
        await rolling_sink.write_from(async_generator(audio_chunk, chunk_duration_ms=200))

    # Verify files were created
    assert output_dir.exists()
    wav_files = list(output_dir.glob(f"{prefix}_*.wav"))
    assert len(wav_files) >= 2  # Should have at least 2 files

    # Verify files have the correct naming pattern
    for wav_file in wav_files:
        assert wav_file.name.startswith(prefix + "_")
        assert wav_file.suffix == ".wav"
        assert wav_file.stat().st_size > 0  # Files should not be empty

    # Clean up
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_rolling_file_sink_segment_timing():
    """Test that files are rolled at approximately the correct duration"""
    output_dir = Path(TEST_ROLLING_DIR)
    prefix = "timing_test_segment"  # Unique prefix
    segment_duration = 0.5  # 0.5 second segments

    # Create test audio (1.5 seconds)
    duration_ms = 1500
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE)

    async with RollingFileSink(
        directory=output_dir,
        prefix=prefix,
        segment_duration_seconds=segment_duration,
        sample_rate=SINE_SAMPLE_RATE,
        channels=1,
    ) as rolling_sink:
        await rolling_sink.write_from(async_generator(audio_chunk, chunk_duration_ms=100))

    # Verify files were created (expect approximately 3 files, but could be 3-4 due to chunk boundaries)
    wav_files = list(output_dir.glob(f"{prefix}_*.wav"))
    assert 3 <= len(wav_files) <= 4  # Should have 3-4 files due to chunk boundary behavior

    # Verify each file (except the last) is approximately the correct duration
    sorted_files = sorted(wav_files)
    for i, wav_file in enumerate(sorted_files[:-1]):  # Exclude last file
        # Read the file to check its duration
        async with LocalFileStreamer(wav_file, chunk_size_ms=100) as streamer:
            total_samples = 0
            async for chunk in streamer.iter_frames():
                total_samples += chunk.samples

            # Calculate duration in seconds
            duration_seconds = total_samples / SINE_SAMPLE_RATE
            # With chunk boundaries, files should be approximately the segment duration
            # (within one chunk duration of tolerance = 100ms = 0.1s)
            assert (
                abs(duration_seconds - segment_duration) <= 0.15
            )  # Allow 150ms tolerance for chunk boundaries

    # Clean up
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_rolling_file_sink_directory_creation():
    """Test that RollingFileSink creates directories automatically"""
    nested_dir = Path(TEST_ROLLING_DIR) / "nested" / "directories"
    prefix = "auto_create_test_dir"  # Unique prefix

    # Ensure the directory doesn't exist
    assert not nested_dir.exists()

    duration_ms = 500
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE)

    async with RollingFileSink(
        directory=nested_dir,
        prefix=prefix,
        segment_duration_seconds=1.0,
        sample_rate=SINE_SAMPLE_RATE,
        channels=1,
    ) as rolling_sink:
        await rolling_sink.write_from(async_generator(audio_chunk))

    # Verify directory was created
    assert nested_dir.exists()
    assert nested_dir.is_dir()

    # Verify file was created
    wav_files = list(nested_dir.glob(f"{prefix}_*.wav"))
    assert len(wav_files) == 1

    # Clean up
    root_test_dir = Path(TEST_ROLLING_DIR)
    if root_test_dir.exists():
        shutil.rmtree(root_test_dir)


@pytest.mark.asyncio
async def test_rolling_file_sink_chunk_based_rolling():
    """Test rolling behavior with individual chunks"""
    output_dir = Path(TEST_ROLLING_DIR)
    prefix = "chunk_test_individual"  # Unique prefix
    segment_duration = 0.1  # Very short segments (100ms)

    # Create multiple small chunks
    chunk_duration_ms = 50  # 50ms chunks
    num_chunks = 5  # Total 250ms, should create 3 files

    async with RollingFileSink(
        directory=output_dir,
        prefix=prefix,
        segment_duration_seconds=segment_duration,
        sample_rate=SINE_SAMPLE_RATE,
        channels=1,
    ) as rolling_sink:
        for i in range(num_chunks):
            chunk = create_sine_wave_audio_chunk(
                chunk_duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE
            )
            await rolling_sink.write(chunk)

    # Verify files were created
    wav_files = list(output_dir.glob(f"{prefix}_*.wav"))
    assert len(wav_files) >= 2  # Should have at least 2 files

    # Clean up
    if output_dir.exists():
        shutil.rmtree(output_dir)


@pytest.mark.asyncio
async def test_rolling_file_sink_error_handling():
    """Test error handling in RollingFileSink"""
    # Test writing without opening
    rolling_sink = RollingFileSink(
        directory=TEST_ROLLING_DIR,
        prefix="error_test",
        segment_duration_seconds=1.0,
        sample_rate=SINE_SAMPLE_RATE,
        channels=1,
    )

    audio_chunk = create_sine_wave_audio_chunk(100, SINE_FREQUENCY, SINE_SAMPLE_RATE)

    with pytest.raises(RuntimeError, match="File sink is not open"):
        await rolling_sink.write(audio_chunk)

    # Test invalid directory (trying to use a file as directory)
    test_file = Path("test_invalid.txt")
    test_file.write_text("test")

    try:
        with pytest.raises(RuntimeError, match="Path exists but is not a directory"):
            async with RollingFileSink(
                directory=test_file,
                prefix="error_test",
                segment_duration_seconds=1.0,
                sample_rate=SINE_SAMPLE_RATE,
                channels=1,
            ):
                pass
    finally:
        test_file.unlink(missing_ok=True)
