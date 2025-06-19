import math
import struct

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


@pytest.mark.asyncio
async def test_resampling_block_channel_conversion_stereo_to_mono():
    """Test converting stereo audio to mono."""
    duration_ms = 500
    sample_rate = 44100

    # Create stereo audio
    stereo_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, sample_rate, channels=2
    )

    # Resample to mono
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_channels=1  # Keep same rate  # Convert to mono
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(stereo_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == sample_rate
        assert chunk.channels == 1  # Should be mono now
        assert chunk.width == stereo_chunk.width
        # Duration should be preserved
        assert (
            abs(
                chunk.milliseconds
                - (len(chunk.audio) / (chunk.rate * chunk.width * chunk.channels) * 1000)
            )
            < 10
        )


@pytest.mark.asyncio
async def test_resampling_block_channel_conversion_mono_to_stereo():
    """Test converting mono audio to stereo."""
    duration_ms = 500
    sample_rate = 44100

    # Create mono audio
    mono_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate, channels=1)

    # Resample to stereo
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_channels=2  # Keep same rate  # Convert to stereo
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(mono_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == sample_rate
        assert chunk.channels == 2  # Should be stereo now
        assert chunk.width == mono_chunk.width
        # Temporal samples should be preserved (mono to stereo doesn't change temporal count)
        expected_samples = mono_chunk.samples  # Same temporal samples, just duplicated to stereo
        total_output_samples = sum(chunk.samples for chunk in output_chunks)
        assert abs(total_output_samples - expected_samples) <= 10


@pytest.mark.asyncio
async def test_resampling_block_width_conversion_16bit_to_8bit():
    """Test converting 16-bit audio to 8-bit."""
    duration_ms = 500
    sample_rate = 44100

    # Create 16-bit audio
    audio_16bit = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate, width=2)

    # Resample to 8-bit
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_width=1  # Keep same rate  # Convert to 8-bit
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_16bit)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == sample_rate
        assert chunk.width == 1  # Should be 8-bit now
        assert chunk.channels == audio_16bit.channels
        # Should have half the byte size (2 bytes -> 1 byte per sample)
        assert len(chunk.audio) == chunk.samples * chunk.channels * 1


@pytest.mark.asyncio
async def test_resampling_block_width_conversion_8bit_to_16bit():
    """Test converting 8-bit audio to 16-bit."""
    duration_ms = 500
    sample_rate = 44100

    # Create 8-bit audio
    audio_8bit = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate, width=1)

    # Resample to 16-bit
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_width=2  # Keep same rate  # Convert to 16-bit
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_8bit)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == sample_rate
        assert chunk.width == 2  # Should be 16-bit now
        assert chunk.channels == audio_8bit.channels
        # Should have double the byte size (1 byte -> 2 bytes per sample)
        assert len(chunk.audio) == chunk.samples * chunk.channels * 2


@pytest.mark.asyncio
async def test_resampling_block_width_conversion_16bit_to_32bit():
    """Test converting 16-bit audio to 32-bit."""
    duration_ms = 500
    sample_rate = 44100

    # Create 16-bit audio
    audio_16bit = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate, width=2)

    # Resample to 32-bit
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_width=4  # Keep same rate  # Convert to 32-bit
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_16bit)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == sample_rate
        assert chunk.width == 4  # Should be 32-bit now
        assert chunk.channels == audio_16bit.channels
        # Should have double the byte size (2 bytes -> 4 bytes per sample)
        assert len(chunk.audio) == chunk.samples * chunk.channels * 4


@pytest.mark.asyncio
async def test_resampling_block_width_conversion_32bit_to_16bit():
    """Test converting 32-bit audio to 16-bit."""
    duration_ms = 500
    sample_rate = 44100

    # Create 32-bit audio
    audio_32bit = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate, width=4)

    # Resample to 16-bit
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_width=2  # Keep same rate  # Convert to 16-bit
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_32bit)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == sample_rate
        assert chunk.width == 2  # Should be 16-bit now
        assert chunk.channels == audio_32bit.channels
        # Should have half the byte size (4 bytes -> 2 bytes per sample)
        assert len(chunk.audio) == chunk.samples * chunk.channels * 2


@pytest.mark.asyncio
async def test_resampling_block_combined_rate_channel_width():
    """Test simultaneous rate, channel, and width conversion."""
    duration_ms = 500
    input_rate = 22050
    output_rate = 44100

    # Create mono 8-bit audio
    input_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, input_rate, channels=1, width=1
    )

    # Convert to stereo 16-bit at higher sample rate
    resampler = ResamplingBlock(
        resample_rate=output_rate,  # Double the rate
        resample_channels=2,  # Mono to stereo
        resample_width=2,  # 8-bit to 16-bit
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(input_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate  # Rate should be changed
        assert chunk.channels == 2  # Should be stereo
        assert chunk.width == 2  # Should be 16-bit

    # Check that total duration is preserved approximately
    total_output_samples = sum(chunk.samples for chunk in output_chunks)
    expected_samples = int(input_chunk.samples * output_rate / input_rate)  # Rate conversion only
    assert abs(total_output_samples - expected_samples) <= 20  # Allow for small discrepancies


@pytest.mark.asyncio
async def test_resampling_block_complex_stereo_to_mono_conversion():
    """Test stereo to mono conversion with different frequencies in each channel."""
    duration_ms = 500
    sample_rate = 44100

    # Create a stereo chunk manually with different frequencies per channel
    num_samples = int(sample_rate * duration_ms / 1000)
    left_freq = 440  # A4
    right_freq = 880  # A5 (octave higher)

    audio_data = bytearray()
    for i in range(num_samples):
        t = i / sample_rate
        # Left channel
        left_sample = int(np.iinfo(np.int16).max * 0.5 * math.sin(2 * math.pi * left_freq * t))
        # Right channel
        right_sample = int(np.iinfo(np.int16).max * 0.5 * math.sin(2 * math.pi * right_freq * t))

        # Interleave left and right samples
        audio_data.extend(struct.pack("<h", left_sample))
        audio_data.extend(struct.pack("<h", right_sample))

    stereo_chunk = AudioChunk(audio=bytes(audio_data), rate=sample_rate, width=2, channels=2)

    # Convert to mono
    resampler = ResamplingBlock(resample_rate=sample_rate, resample_channels=1)

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(stereo_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.channels == 1
        # The mono output should be the average of both channels
        # We can't easily verify the exact content, but we can verify format
        assert chunk.rate == sample_rate
        assert chunk.width == 2


@pytest.mark.asyncio
async def test_resampling_block_unsupported_channel_conversion():
    """Test handling of unsupported channel conversions."""
    duration_ms = 200
    sample_rate = 44100

    # Create stereo audio
    stereo_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, sample_rate, channels=2
    )

    # Try to convert to 3-channel audio (unsupported)
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_channels=3  # Unsupported conversion
    )

    output_chunks = []
    # This should not crash, but might log a warning
    async for output_chunk in resampler.process(async_generator(stereo_chunk)):
        output_chunks.append(output_chunk)

    # Should still produce output, but channels might not be converted
    assert len(output_chunks) > 0


@pytest.mark.asyncio
async def test_resampling_block_no_conversion_needed():
    """Test that no conversion is applied when input format matches target format."""
    duration_ms = 500
    sample_rate = 44100
    channels = 2
    width = 2

    audio_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, sample_rate, channels=channels, width=width
    )

    # Configure resampler with same format as input
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_channels=channels, resample_width=width
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0

    # Since no conversion is needed, output should be identical to input
    total_input_bytes = len(audio_chunk.audio)
    total_output_bytes = sum(len(chunk.audio) for chunk in output_chunks)
    assert total_input_bytes == total_output_bytes

    for chunk in output_chunks:
        assert chunk.rate == sample_rate
        assert chunk.channels == channels
        assert chunk.width == width


@pytest.mark.asyncio
async def test_resampling_block_process_chunk_with_conversions():
    """Test process_chunk method with channel and width conversions."""
    duration_ms = 100
    input_rate = 22050
    output_rate = 44100

    # Create mono 8-bit audio
    audio_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, input_rate, channels=1, width=1
    )

    # Convert to stereo 16-bit at higher rate
    resampler = ResamplingBlock(resample_rate=output_rate, resample_channels=2, resample_width=2)
    await resampler.open()

    output_chunks = []
    async for output_chunk in resampler.process_chunk(audio_chunk):
        output_chunks.append(output_chunk)

    await resampler.close()

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate
        assert chunk.channels == 2
        assert chunk.width == 2


@pytest.mark.asyncio
async def test_resampling_block_buffering_consistent_samples():
    """Test that buffering maintains consistent temporal sample counts."""
    duration_ms = 1000  # 1 second
    input_rate = 44100
    output_rate = 44100  # Keep same rate to focus on format conversion

    # Create stereo 32-bit audio
    input_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, input_rate, channels=2, width=4
    )

    # Convert to mono 16-bit (both channel and width conversion)
    resampler = ResamplingBlock(
        resample_rate=output_rate,
        resample_channels=1,
        resample_width=2,
        target_samples_per_chunk=1024,  # Fixed chunk size for consistent output
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(input_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0

    # All chunks except possibly the last should have exactly 1024 samples
    for i, chunk in enumerate(output_chunks[:-1]):
        assert chunk.samples == 1024, f"Chunk {i} has {chunk.samples} samples, expected 1024"
        assert chunk.channels == 1
        assert chunk.width == 2
        assert chunk.rate == output_rate

    # Last chunk can be smaller
    last_chunk = output_chunks[-1]
    assert last_chunk.samples <= 1024
    assert last_chunk.channels == 1
    assert last_chunk.width == 2
    assert last_chunk.rate == output_rate


@pytest.mark.asyncio
async def test_resampling_block_temporal_duration_preservation():
    """Test that temporal duration is preserved across format conversions."""
    duration_ms = 500
    sample_rate = 48000

    # Create mono 8-bit audio
    input_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, sample_rate, channels=1, width=1
    )

    # Convert to stereo 32-bit
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_channels=2, resample_width=4  # Keep same rate
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(input_chunk)):
        output_chunks.append(output_chunk)

    # Calculate total duration from output chunks
    total_output_duration_ms = 0
    for chunk in output_chunks:
        chunk_duration_ms = (chunk.samples / chunk.rate) * 1000
        total_output_duration_ms += chunk_duration_ms

    # Duration should be preserved (within small tolerance)
    assert (
        abs(total_output_duration_ms - duration_ms) < 10
    ), f"Expected {duration_ms}ms, got {total_output_duration_ms:.2f}ms"


@pytest.mark.asyncio
async def test_resampling_block_buffer_prevents_sample_loss():
    """Test that buffering prevents sample loss during conversions."""
    duration_ms = 300
    sample_rate = 22050

    # Create test audio
    input_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, sample_rate, channels=2, width=2
    )

    input_temporal_samples = input_chunk.samples  # AudioChunk.samples is already temporal

    # Convert stereo to mono (should preserve temporal samples)
    resampler = ResamplingBlock(resample_rate=sample_rate, resample_channels=1, resample_width=2)

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(input_chunk)):
        output_chunks.append(output_chunk)

    # Calculate total temporal samples from output
    total_output_temporal_samples = sum(chunk.samples for chunk in output_chunks)

    # Should preserve the same number of temporal samples
    assert (
        abs(total_output_temporal_samples - input_temporal_samples) <= 1
    ), f"Expected ~{input_temporal_samples} temporal samples, got {total_output_temporal_samples}"


@pytest.mark.asyncio
async def test_resampling_block_multiple_small_chunks():
    """Test buffering with multiple small input chunks."""
    duration_ms = 100  # Small chunks
    sample_rate = 44100
    num_chunks = 5

    # Create multiple small chunks
    input_chunks = [
        create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate, channels=2, width=4)
        for _ in range(num_chunks)
    ]

    # Convert to mono 16-bit with fixed output chunk size
    resampler = ResamplingBlock(
        resample_rate=sample_rate,
        resample_channels=1,
        resample_width=2,
        target_samples_per_chunk=2048,
    )

    async def multi_chunk_generator():
        for chunk in input_chunks:
            yield chunk

    output_chunks = []
    async for output_chunk in resampler.process(multi_chunk_generator()):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0

    # Most chunks should have exactly 2048 samples due to buffering
    consistent_chunks = [chunk for chunk in output_chunks if chunk.samples == 2048]
    assert len(consistent_chunks) >= 1, "Should have at least one consistently sized chunk"

    # All chunks should have correct format
    for chunk in output_chunks:
        assert chunk.channels == 1
        assert chunk.width == 2
        assert chunk.rate == sample_rate


@pytest.mark.asyncio
async def test_resampling_block_width_conversion_sample_alignment():
    """Test that width conversion maintains proper sample alignment."""
    duration_ms = 200
    sample_rate = 48000

    # Test 32-bit to 16-bit conversion
    input_32bit = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, sample_rate, channels=1, width=4
    )

    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_channels=1, resample_width=2  # 32-bit to 16-bit
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(input_32bit)):
        output_chunks.append(output_chunk)

    # Total samples should be preserved
    input_samples = input_32bit.samples
    total_output_samples = sum(chunk.samples for chunk in output_chunks)

    assert (
        abs(total_output_samples - input_samples) <= 1
    ), f"Sample count mismatch: input={input_samples}, output={total_output_samples}"

    # Verify all output is 16-bit
    for chunk in output_chunks:
        assert chunk.width == 2
        assert chunk.channels == 1
        assert chunk.rate == sample_rate


@pytest.mark.asyncio
async def test_resampling_block_channel_conversion_sample_alignment():
    """Test that channel conversion maintains proper temporal alignment."""
    duration_ms = 250
    sample_rate = 44100

    # Test stereo to mono conversion
    stereo_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, sample_rate, channels=2, width=2
    )

    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_channels=1, resample_width=2  # Stereo to mono
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(stereo_chunk)):
        output_chunks.append(output_chunk)

    # Temporal samples should be preserved (AudioChunk.samples is already temporal)
    input_temporal_samples = stereo_chunk.samples
    total_output_samples = sum(chunk.samples for chunk in output_chunks)

    assert (
        abs(total_output_samples - input_temporal_samples) <= 1
    ), f"Temporal sample mismatch: expected~{input_temporal_samples}, got {total_output_samples}"

    # Verify all output is mono
    for chunk in output_chunks:
        assert chunk.channels == 1
        assert chunk.width == 2
        assert chunk.rate == sample_rate
