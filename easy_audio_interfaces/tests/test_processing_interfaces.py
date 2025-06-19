import numpy as np
import pytest
from wyoming.audio import AudioChunk

from easy_audio_interfaces import RechunkingBlock, ResamplingBlock

from .utils import async_generator, create_sine_wave_audio_chunk

SINE_FREQUENCY = 440
SINE_SAMPLE_RATE = 44100


@pytest.mark.asyncio
async def test_rechunking_block_in_ms():
    # Create a sample AudioChunk of 10 seconds duration
    duration_ms = 10000  # 10 seconds
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE)

    # Initialize RechunkingBlock with chunk size 500ms
    chunk_size_ms = 500
    rechunker = RechunkingBlock(chunk_size_ms=chunk_size_ms)

    # Process the frame
    output_chunks = []
    async for output_chunk in rechunker.process(async_generator(audio_chunk)):
        output_chunks.append(output_chunk)

    # Check that we have the expected number of chunks
    num_expected_chunks = 10000 // chunk_size_ms

    # should be 20 chunks of 500 ms
    assert len(output_chunks) == num_expected_chunks
    for i, chunk in enumerate(output_chunks):
        expected_ms = chunk_size_ms
        assert abs(chunk.milliseconds - expected_ms) < 50  # Allow small discrepancy


@pytest.mark.asyncio
async def test_rechunking_block_in_samples():
    # Create a sample AudioChunk of 10 seconds duration
    duration_ms = 10000  # 10 seconds
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, SINE_SAMPLE_RATE)

    # Initialize RechunkingBlock with chunk size 512 samples
    chunk_size_samples = 512
    rechunker = RechunkingBlock(chunk_size_samples=chunk_size_samples)

    # Process the frame
    output_chunks = []
    async for output_chunk in rechunker.process(async_generator(audio_chunk)):
        output_chunks.append(output_chunk)

    # Check that we have expected number of chunks
    num_expected_chunks = (duration_ms // 1000) * SINE_SAMPLE_RATE // chunk_size_samples

    assert (
        abs(len(output_chunks) - num_expected_chunks) <= 1  # Last chunk may be fractional
    ), f"Expected {num_expected_chunks} chunks, got {len(output_chunks)}"
    for i, chunk in enumerate(output_chunks[:-2]):  # Last chunk may be smaller
        assert (
            chunk.samples == chunk_size_samples
        ), f"Chunk {i} has {chunk.samples} samples, total chunks: {len(output_chunks)}"


@pytest.mark.asyncio
async def test_resampling_block_basic():
    """Test basic resampling functionality from 44100 to 48000 Hz."""
    # Create a sample AudioChunk at 44100 Hz
    duration_ms = 1000  # 1 second
    input_rate = 44100
    output_rate = 48000
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, input_rate)

    # Initialize ResamplingBlock
    resampler = ResamplingBlock(resample_rate=output_rate)

    # Process the audio
    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_chunk)):
        output_chunks.append(output_chunk)

    # Verify the output
    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate
        assert chunk.width == audio_chunk.width
        assert chunk.channels == audio_chunk.channels

    # Check that duration is preserved (approximately)
    total_output_samples = sum(chunk.samples for chunk in output_chunks)
    expected_samples = int(audio_chunk.samples * output_rate / input_rate)
    assert abs(total_output_samples - expected_samples) <= 10  # Allow small discrepancy


@pytest.mark.asyncio
async def test_resampling_block_no_change():
    """Test that resampling with same rate passes through unchanged."""
    duration_ms = 500
    sample_rate = 44100
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate)

    # Initialize ResamplingBlock with same rate
    resampler = ResamplingBlock(resample_rate=sample_rate)

    # Process the audio
    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_chunk)):
        output_chunks.append(output_chunk)

    # Should be passed through unchanged
    assert len(output_chunks) > 0
    for i, chunk in enumerate(output_chunks):
        assert chunk.rate == sample_rate
        assert chunk.width == audio_chunk.width
        assert chunk.channels == audio_chunk.channels


@pytest.mark.asyncio
async def test_resampling_block_downsampling():
    """Test downsampling from 48000 to 22050 Hz."""
    duration_ms = 1000
    input_rate = 48000
    output_rate = 22050
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, input_rate)

    resampler = ResamplingBlock(resample_rate=output_rate)

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate

    # Check sample count
    total_output_samples = sum(chunk.samples for chunk in output_chunks)
    expected_samples = int(audio_chunk.samples * output_rate / input_rate)
    assert abs(total_output_samples - expected_samples) <= 10


@pytest.mark.asyncio
async def test_resampling_block_different_qualities():
    """Test different quality settings."""
    duration_ms = 500
    input_rate = 44100
    output_rate = 48000
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, input_rate)

    qualities = ["HQ", "VHQ", "MQ", "LQ", "QQ"]

    for quality in qualities:
        resampler = ResamplingBlock(resample_rate=output_rate, quality=quality)

        output_chunks = []
        async for output_chunk in resampler.process(async_generator(audio_chunk)):
            output_chunks.append(output_chunk)

        assert len(output_chunks) > 0
        for chunk in output_chunks:
            assert chunk.rate == output_rate


@pytest.mark.asyncio
async def test_resampling_block_stereo():
    """Test resampling with stereo audio."""
    duration_ms = 500
    input_rate = 44100
    output_rate = 48000
    channels = 2  # Stereo
    audio_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, input_rate, channels=channels
    )

    resampler = ResamplingBlock(resample_rate=output_rate)

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate
        assert chunk.channels == channels
        assert chunk.width == audio_chunk.width


@pytest.mark.asyncio
async def test_resampling_block_8bit_audio():
    """Test resampling with 8-bit audio."""
    duration_ms = 500
    input_rate = 22050
    output_rate = 44100

    # Create 8-bit audio chunk
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, input_rate, width=1)

    resampler = ResamplingBlock(resample_rate=output_rate)

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate
        assert chunk.width == 1  # Should preserve 8-bit width
        assert chunk.channels == audio_chunk.channels


@pytest.mark.asyncio
async def test_resampling_block_32bit_int():
    """Test resampling with 32-bit int audio."""
    duration_ms = 500
    input_rate = 44100
    output_rate = 48000

    # Create a simple int32 audio chunk
    num_samples = int(input_rate * duration_ms / 1000)
    samples = np.sin(2 * np.pi * SINE_FREQUENCY * np.arange(num_samples) / input_rate)
    audio_data = (samples * np.iinfo(np.int32).max).astype(np.int32).tobytes()

    audio_chunk = AudioChunk(
        audio=audio_data, rate=input_rate, width=4, channels=1  # 32-bit = 4 bytes
    )

    resampler = ResamplingBlock(resample_rate=output_rate)

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate
        assert chunk.width == 4  # Should preserve 32-bit width
        assert chunk.channels == 1


@pytest.mark.asyncio
async def test_resampling_block_extreme_ratios():
    """Test resampling with extreme rate ratios."""
    duration_ms = 200

    # Test very high upsampling
    audio_chunk_low = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, 8000)
    resampler_up = ResamplingBlock(resample_rate=96000)

    output_chunks = []
    async for output_chunk in resampler_up.process(async_generator(audio_chunk_low)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    assert all(chunk.rate == 96000 for chunk in output_chunks)

    # Test very high downsampling
    audio_chunk_high = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, 96000)
    resampler_down = ResamplingBlock(resample_rate=8000)

    output_chunks = []
    async for output_chunk in resampler_down.process(async_generator(audio_chunk_high)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    assert all(chunk.rate == 8000 for chunk in output_chunks)


# New tests for process_chunk functionality
@pytest.mark.asyncio
async def test_resampling_block_process_chunk():
    """Test the new process_chunk method with ResamplingBlock."""
    duration_ms = 100
    input_rate = 16000
    output_rate = 48000
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, input_rate)

    resampler = ResamplingBlock(resample_rate=output_rate)
    await resampler.open()

    # Test process_chunk method
    output_chunks = []
    async for output_chunk in resampler.process_chunk(audio_chunk):
        output_chunks.append(output_chunk)

    await resampler.close()

    # Verify results
    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate
        assert chunk.width == audio_chunk.width
        assert chunk.channels == audio_chunk.channels

    # Check that total samples are approximately correct
    total_output_samples = sum(chunk.samples for chunk in output_chunks)
    expected_samples = int(audio_chunk.samples * output_rate / input_rate)
    assert abs(total_output_samples - expected_samples) <= 10


@pytest.mark.asyncio
async def test_resampling_block_process_chunk_stereo():
    """Test process_chunk with stereo audio."""
    duration_ms = 100
    input_rate = 44100
    output_rate = 48000
    channels = 2
    audio_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, input_rate, channels=channels
    )

    resampler = ResamplingBlock(resample_rate=output_rate)
    await resampler.open()

    output_chunks = []
    async for output_chunk in resampler.process_chunk(audio_chunk):
        output_chunks.append(output_chunk)

    await resampler.close()

    # Verify stereo is preserved
    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate
        assert chunk.channels == channels
        assert chunk.width == audio_chunk.width


@pytest.mark.asyncio
async def test_rechunking_block_process_chunk():
    """Test the new process_chunk method with RechunkingBlock."""
    duration_ms = 200  # 200ms of audio
    sample_rate = 48000
    chunk_size_ms = 50  # Split into 50ms chunks
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate)

    rechunker = RechunkingBlock(chunk_size_ms=chunk_size_ms)
    await rechunker.open()

    # Test process_chunk method
    output_chunks = []
    async for output_chunk in rechunker.process_chunk(audio_chunk):
        output_chunks.append(output_chunk)

    await rechunker.close()

    # Should produce 4 chunks of 50ms each
    assert len(output_chunks) == 4
    for chunk in output_chunks:
        assert chunk.rate == sample_rate
        assert chunk.width == audio_chunk.width
        assert chunk.channels == audio_chunk.channels
        # Each chunk should be approximately 50ms
        chunk_duration_ms = chunk.samples / chunk.rate * 1000
        assert abs(chunk_duration_ms - chunk_size_ms) < 5  # Allow small tolerance


@pytest.mark.asyncio
async def test_process_chunk_vs_process_consistency():
    """Test that process_chunk produces the same results as process for single chunks."""
    duration_ms = 100
    input_rate = 22050
    output_rate = 44100
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, input_rate)

    resampler = ResamplingBlock(resample_rate=output_rate)
    await resampler.open()

    # Test process_chunk
    process_chunk_results = []
    async for output_chunk in resampler.process_chunk(audio_chunk):
        process_chunk_results.append(output_chunk)

    # Test process with single-item generator
    process_results = []
    async for output_chunk in resampler.process(async_generator(audio_chunk)):
        process_results.append(output_chunk)

    await resampler.close()

    # Results should be identical
    assert len(process_chunk_results) == len(process_results)
    for chunk1, chunk2 in zip(process_chunk_results, process_results):
        assert chunk1.rate == chunk2.rate
        assert chunk1.width == chunk2.width
        assert chunk1.channels == chunk2.channels
        assert len(chunk1.audio) == len(chunk2.audio)
        # Audio data should be identical (or very close due to floating point)
        assert chunk1.audio == chunk2.audio
