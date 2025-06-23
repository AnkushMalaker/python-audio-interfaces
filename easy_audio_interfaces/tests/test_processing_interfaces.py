import math
import struct
import tempfile
import wave
from pathlib import Path
from typing import Union

import numpy as np
import pytest
from scipy.signal import correlate
from wyoming.audio import AudioChunk

from easy_audio_interfaces import RechunkingBlock, ResamplingBlock

from .utils import async_generator, create_sine_wave_audio_chunk

SINE_FREQUENCY = 440
SINE_SAMPLE_RATE = 44100


def calculate_relative_tolerance(expected: int, relative_tolerance: float = 0.0005) -> int:
    """Calculate a relative tolerance for sample count assertions."""
    return max(1, int(relative_tolerance * expected))


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
    tolerance = calculate_relative_tolerance(expected_samples)
    assert abs(total_output_samples - expected_samples) <= tolerance


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
    tolerance = calculate_relative_tolerance(expected_samples)
    assert abs(total_output_samples - expected_samples) <= tolerance


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
async def test_resampling_block_unsupported_width_input_audio():
    """Test that unsupported input audio width raises NotImplementedError."""
    duration_ms = 500
    input_rate = 22050
    output_rate = 44100

    # Create 8-bit audio chunk (unsupported)
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, input_rate, width=1)

    resampler = ResamplingBlock(resample_rate=output_rate)

    # Should raise NotImplementedError for unsupported width
    with pytest.raises(
        NotImplementedError, match="Only 16-bit and 32-bit audio processing is supported"
    ):
        async for output_chunk in resampler.process(async_generator(audio_chunk)):
            pass


@pytest.mark.asyncio
async def test_resampling_block_unsupported_width_output_audio():
    """Test that unsupported output audio width raises NotImplementedError."""
    duration_ms = 500
    input_rate = 22050
    output_rate = 44100

    # Create 16-bit audio chunk
    audio_chunk = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, input_rate, width=2)

    resampler = ResamplingBlock(
        resample_rate=output_rate, resample_width=1
    )  # Convert to 8-bit (unsupported)

    # Should raise NotImplementedError for unsupported output width
    with pytest.raises(
        NotImplementedError, match="Only 16-bit and 32-bit audio processing is supported"
    ):
        async for output_chunk in resampler.process(async_generator(audio_chunk)):
            pass


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
    tolerance = calculate_relative_tolerance(expected_samples)
    assert abs(total_output_samples - expected_samples) <= tolerance


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

    # Reset resampler state since process_chunk finalizes the ResampleStream
    await resampler.close()

    resampler = ResamplingBlock(resample_rate=output_rate)
    await resampler.open()

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
    tolerance = calculate_relative_tolerance(expected_samples)
    assert abs(total_output_samples - expected_samples) <= tolerance


@pytest.mark.asyncio
async def test_resampling_block_width_conversion_16bit_to_8bit():
    """Test that converting to 8-bit raises NotImplementedError."""
    duration_ms = 500
    sample_rate = 44100

    # Create 16-bit audio
    audio_16bit = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate, width=2)

    # Try to resample to 8-bit
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_width=1  # Keep same rate  # Convert to 8-bit
    )

    # Should raise NotImplementedError for 8-bit target
    with pytest.raises(
        NotImplementedError, match="Only 16-bit and 32-bit audio processing is supported"
    ):
        async for output_chunk in resampler.process(async_generator(audio_16bit)):
            pass


@pytest.mark.asyncio
async def test_resampling_block_width_conversion_8bit_to_16bit():
    """Test that converting from 8-bit raises NotImplementedError."""
    duration_ms = 500
    sample_rate = 44100

    # Create 8-bit audio
    audio_8bit = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate, width=1)

    # Try to resample to 16-bit
    resampler = ResamplingBlock(
        resample_rate=sample_rate, resample_width=2  # Keep same rate  # Convert to 16-bit
    )

    # Should raise NotImplementedError for 8-bit input
    with pytest.raises(
        NotImplementedError, match="Only 16-bit and 32-bit audio processing is supported"
    ):
        async for output_chunk in resampler.process(async_generator(audio_8bit)):
            pass


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

    # Create mono 16-bit audio (changed from 8-bit)
    input_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, input_rate, channels=1, width=2
    )

    # Convert to stereo 32-bit at higher sample rate
    resampler = ResamplingBlock(
        resample_rate=output_rate,  # Double the rate
        resample_channels=2,  # Mono to stereo
        resample_width=4,  # 16-bit to 32-bit
    )

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(input_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate  # Rate should be changed
        assert chunk.channels == 2  # Should be stereo
        assert chunk.width == 4  # Should be 32-bit

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

    # Should raise ValueError for unsupported channel conversion
    with pytest.raises(ValueError, match="Unsupported channel conversion"):
        async for output_chunk in resampler.process(async_generator(stereo_chunk)):
            pass


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

    # Create mono 16-bit audio
    audio_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, input_rate, channels=1, width=2
    )

    # Convert to stereo 32-bit at higher rate
    resampler = ResamplingBlock(resample_rate=output_rate, resample_channels=2, resample_width=4)
    await resampler.open()

    output_chunks = []
    async for output_chunk in resampler.process_chunk(audio_chunk):
        output_chunks.append(output_chunk)

    await resampler.close()

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == output_rate
        assert chunk.channels == 2
        assert chunk.width == 4


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
    resampler = ResamplingBlock(resample_rate=output_rate, resample_channels=1, resample_width=2)

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(input_chunk)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0

    # All chunks should have correct format and preserve temporal samples
    total_output_samples = sum(chunk.samples for chunk in output_chunks)

    for chunk in output_chunks:
        assert chunk.channels == 1
        assert chunk.width == 2
        assert chunk.rate == output_rate

    # Total temporal samples should be preserved
    input_temporal_samples = input_chunk.samples
    assert abs(total_output_samples - input_temporal_samples) <= 1


@pytest.mark.asyncio
async def test_resampling_block_temporal_duration_preservation():
    """Test that temporal duration is preserved across format conversions."""
    duration_ms = 500
    sample_rate = 48000

    # Create mono 16-bit audio
    input_chunk = create_sine_wave_audio_chunk(
        duration_ms, SINE_FREQUENCY, sample_rate, channels=1, width=2
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

    # Convert to mono 16-bit
    resampler = ResamplingBlock(
        resample_rate=sample_rate,
        resample_channels=1,
        resample_width=2,
    )

    async def multi_chunk_generator():
        for chunk in input_chunks:
            yield chunk

    output_chunks = []
    async for output_chunk in resampler.process(multi_chunk_generator()):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0

    # Verify temporal samples are preserved across all input chunks
    total_input_samples = sum(chunk.samples for chunk in input_chunks)
    total_output_samples = sum(chunk.samples for chunk in output_chunks)
    assert abs(total_output_samples - total_input_samples) <= len(
        input_chunks
    )  # Allow small tolerance per chunk

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


def test_resampling_block_quality_validation():
    """Test that invalid quality parameters raise ValueError."""
    with pytest.raises(ValueError, match="quality must be one of"):
        ResamplingBlock(16000, quality="INVALID")

    # Test all valid qualities work
    valid_qualities = ["HQ", "VHQ", "MQ", "LQ", "QQ"]
    for quality in valid_qualities:
        resampler = ResamplingBlock(16000, quality=quality)
        assert resampler._quality == quality


def test_resampling_block_int32_to_int16_shift_mode():
    """Test 32-bit to 16-bit conversion using shift mode."""
    resampler = ResamplingBlock(16000)

    # Test data with known values that fit properly in int32
    # Using values that when shifted represent proper 16-bit values
    test_data = np.array(
        [
            0,  # 0 >> 16 = 0
            1000 << 16,  # 1000 << 16, then >> 16 = 1000
            -1000 << 16,  # -1000 << 16, then >> 16 = -1000
            32767 << 16,  # 32767 << 16, then >> 16 = 32767 (max int16)
        ],
        dtype=np.int32,
    )
    result = resampler._int32_to_int16(test_data)

    expected = np.array([0, 1000, -1000, 32767], dtype=np.int16)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int16


@pytest.mark.asyncio
async def test_resampling_block_32bit_to_16bit_shift_mode_integration():
    """Test full integration of 32-bit to 16-bit conversion with shift mode."""
    duration_ms = 100
    sample_rate = 44100

    # Create 32-bit audio
    audio_32bit = create_sine_wave_audio_chunk(duration_ms, SINE_FREQUENCY, sample_rate, width=4)

    # Resample to 16-bit using shift mode
    resampler = ResamplingBlock(resample_rate=sample_rate, resample_width=2)

    output_chunks = []
    async for output_chunk in resampler.process(async_generator(audio_32bit)):
        output_chunks.append(output_chunk)

    assert len(output_chunks) > 0
    for chunk in output_chunks:
        assert chunk.rate == sample_rate
        assert chunk.width == 2  # Should be 16-bit now
        assert chunk.channels == audio_32bit.channels


# Property-based tests for round-trip accuracy

# Common sample rates used in real-world applications
COMMON_SAMPLE_RATES = [8000, 16000, 22050, 44100, 48000]


def save_audio_to_wav(
    audio_data: bytes, sample_rate: int, width: int, channels: int, filepath: Path
) -> None:
    """Save audio data to a WAV file."""
    with wave.open(str(filepath), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)


def load_audio_from_wav(filepath: Path) -> tuple[bytes, int, int, int]:
    """Load audio data from a WAV file."""
    with wave.open(str(filepath), "rb") as wav_file:
        channels = wav_file.getnchannels()
        width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        audio_data = wav_file.readframes(wav_file.getnframes())
    return audio_data, sample_rate, width, channels


def align_signals(ref: np.ndarray, test: np.ndarray, max_search: int = 4096):
    """
    Align `test` to `ref` by maximising cross-correlation.
    Returns two arrays of equal length.
    """
    # Normalise to avoid overflow in int32 ↔ float mixes
    ref_f = ref.astype(np.float64)
    test_f = test.astype(np.float64)

    # Search only a reasonable window to keep it fast
    lag_corr = correlate(test_f[:max_search], ref_f[:max_search], mode="full")
    lag = np.argmax(lag_corr) - (len(ref_f[:max_search]) - 1)

    if lag > 0:
        test_f = test_f[lag:]
        ref_f = ref_f[: len(test_f)]
    elif lag < 0:
        ref_f = ref_f[-lag:]
        test_f = test_f[: len(ref_f)]

    return ref_f, test_f


def generate_test_signal(
    duration_samples: int, sample_rate: int, dtype: Union[type[np.int16], type[np.int32]]
) -> np.ndarray:
    """Generate a complex test signal with multiple frequency components."""
    time = np.arange(duration_samples) / sample_rate

    # Multi-tone signal with different frequencies and amplitudes
    # Use frequencies that are well within the Nyquist frequency for most sample rates
    base_freq = min(440, sample_rate // 8)  # Ensure we don't go too high
    signal = (
        0.4 * np.sin(2 * np.pi * base_freq * time)  # Base frequency
        + 0.3 * np.sin(2 * np.pi * base_freq * 2 * time)  # Harmonic
        + 0.2 * np.sin(2 * np.pi * base_freq * 3 * time)  # Second harmonic
        + 0.1 * np.sin(2 * np.pi * base_freq * 0.5 * time)  # Sub-harmonic
    )

    # Scale to appropriate range for the data type
    if dtype == np.int16:
        max_val = np.iinfo(np.int16).max
    elif dtype == np.int32:
        max_val = np.iinfo(np.int32).max
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Scale to 70% of max to avoid clipping
    scaled_signal = (signal * 0.7 * max_val).astype(dtype)
    return scaled_signal


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate", COMMON_SAMPLE_RATES)
@pytest.mark.parametrize("original_width", [2, 4])
@pytest.mark.parametrize("intermediate_width", [2, 4])
@pytest.mark.parametrize("duration_ms", [300, 500])
async def test_round_trip_width_conversion_wav(
    sample_rate, original_width, intermediate_width, duration_ms
):
    """Test round-trip width conversion accuracy using common sample rates and WAV files."""
    # Skip if widths are the same (no conversion)
    if original_width == intermediate_width:
        pytest.skip("No width conversion needed")

    # Generate test signal
    num_samples = int(sample_rate * duration_ms / 1000)
    original_dtype = np.int16 if original_width == 2 else np.int32
    original_signal = generate_test_signal(num_samples, sample_rate, original_dtype)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save original audio to WAV
        original_wav = temp_path / "original.wav"
        save_audio_to_wav(original_signal.tobytes(), sample_rate, original_width, 1, original_wav)

        # Load back from WAV to verify round-trip through file format
        loaded_audio, loaded_rate, loaded_width, loaded_channels = load_audio_from_wav(original_wav)

        # Create AudioChunk from loaded data
        original_chunk = AudioChunk(
            audio=loaded_audio, rate=loaded_rate, width=loaded_width, channels=loaded_channels
        )

        # First conversion: original_width -> intermediate_width
        resampler1 = ResamplingBlock(
            resample_rate=sample_rate,
            resample_width=intermediate_width,
        )
        await resampler1.open()

        intermediate_chunks = []
        async for chunk in resampler1.process_chunk(original_chunk):
            intermediate_chunks.append(chunk)

        await resampler1.close()

        # Combine intermediate chunks and save to WAV
        intermediate_audio = b"".join(chunk.audio for chunk in intermediate_chunks)
        intermediate_wav = temp_path / "intermediate.wav"
        save_audio_to_wav(intermediate_audio, sample_rate, intermediate_width, 1, intermediate_wav)

        # Load intermediate from WAV
        loaded_intermediate, _, _, _ = load_audio_from_wav(intermediate_wav)
        intermediate_chunk = AudioChunk(
            audio=loaded_intermediate, rate=sample_rate, width=intermediate_width, channels=1
        )

        # Second conversion: intermediate_width -> original_width (round trip)
        resampler2 = ResamplingBlock(
            resample_rate=sample_rate,
            resample_width=original_width,
        )
        await resampler2.open()

        final_chunks = []
        async for chunk in resampler2.process_chunk(intermediate_chunk):
            final_chunks.append(chunk)

        await resampler2.close()

        # Combine final chunks and save to WAV
        final_audio = b"".join(chunk.audio for chunk in final_chunks)
        final_wav = temp_path / "final.wav"
        save_audio_to_wav(final_audio, sample_rate, original_width, 1, final_wav)

        # Load final from WAV
        loaded_final, _, _, _ = load_audio_from_wav(final_wav)

        # Compare original and final
        original_array = np.frombuffer(loaded_audio, dtype=original_dtype)
        final_array = np.frombuffer(loaded_final, dtype=original_dtype)

        # Ensure same length for comparison
        min_len = min(len(original_array), len(final_array))
        original_array = original_array[:min_len]
        final_array = final_array[:min_len]

        # For width conversion with "shift" mode:
        # - 16->32->16 should be perfect (no precision loss)
        # - 32->16->32 will lose precision (lower 16 bits discarded)
        if original_width == 2 and intermediate_width == 4:
            # 16->32->16 should be perfect
            assert np.array_equal(
                original_array, final_array
            ), "16->32->16 conversion should be lossless"
        else:
            # 32->16->32 loses precision, but should be very close
            max_error = np.max(
                np.abs(original_array.astype(np.int64) - final_array.astype(np.int64))
            )
            # Allow for precision loss in lower 16 bits
            max_allowed_error = 2**16  # Error from losing lower 16 bits
            assert (
                max_error <= max_allowed_error
            ), f"32->16->32 conversion error {max_error} exceeds expected {max_allowed_error}"


@pytest.mark.asyncio
@pytest.mark.parametrize("original_rate", COMMON_SAMPLE_RATES)
@pytest.mark.parametrize("intermediate_rate", COMMON_SAMPLE_RATES)
@pytest.mark.parametrize("original_width", [2, 4])
@pytest.mark.parametrize("intermediate_width", [2, 4])
@pytest.mark.parametrize("duration_ms", [300, 500])
async def test_round_trip_combined_conversion_wav(
    original_rate, intermediate_rate, original_width, intermediate_width, duration_ms
):
    """Test round-trip combined rate and width conversion accuracy using common sample rates and WAV files."""
    # Skip if no meaningful conversion
    if (
        abs(original_rate - intermediate_rate) <= original_rate * 0.05
        and original_width == intermediate_width
    ):
        pytest.skip("No meaningful conversion needed")

    # Generate test signal
    num_samples = int(original_rate * duration_ms / 1000)
    original_dtype = np.int16 if original_width == 2 else np.int32
    original_signal = generate_test_signal(num_samples, original_rate, original_dtype)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Save original audio to WAV
        original_wav = temp_path / "original.wav"
        save_audio_to_wav(original_signal.tobytes(), original_rate, original_width, 1, original_wav)

        # Load back from WAV to verify round-trip through file format
        loaded_audio, loaded_rate, loaded_width, loaded_channels = load_audio_from_wav(original_wav)

        # Create AudioChunk from loaded data
        original_chunk = AudioChunk(
            audio=loaded_audio, rate=loaded_rate, width=loaded_width, channels=loaded_channels
        )

        # First conversion: original -> intermediate
        resampler1 = ResamplingBlock(
            resample_rate=intermediate_rate,
            resample_width=intermediate_width,
        )
        await resampler1.open()

        intermediate_chunks = []
        async for chunk in resampler1.process_chunk(original_chunk):
            intermediate_chunks.append(chunk)

        await resampler1.close()

        # Combine intermediate chunks and save to WAV
        intermediate_audio = b"".join(chunk.audio for chunk in intermediate_chunks)
        intermediate_wav = temp_path / "intermediate.wav"
        save_audio_to_wav(
            intermediate_audio, intermediate_rate, intermediate_width, 1, intermediate_wav
        )

        # Load intermediate from WAV
        loaded_intermediate, _, _, _ = load_audio_from_wav(intermediate_wav)
        intermediate_chunk = AudioChunk(
            audio=loaded_intermediate, rate=intermediate_rate, width=intermediate_width, channels=1
        )

        # Second conversion: intermediate -> original (round trip)
        resampler2 = ResamplingBlock(resample_rate=original_rate, resample_width=original_width)
        await resampler2.open()

        final_chunks = []
        async for chunk in resampler2.process_chunk(intermediate_chunk):
            final_chunks.append(chunk)

        await resampler2.close()

        # Combine final chunks and save to WAV
        final_audio = b"".join(chunk.audio for chunk in final_chunks)
        final_wav = temp_path / "final.wav"
        save_audio_to_wav(final_audio, original_rate, original_width, 1, final_wav)

        # Load final from WAV
        loaded_final, _, _, _ = load_audio_from_wav(final_wav)

        # Compare original and final
        original_array = np.frombuffer(loaded_audio, dtype=original_dtype)
        final_array = np.frombuffer(loaded_final, dtype=original_dtype)

        # Align signals to compensate for resampler group delay
        aligned_ref, aligned_test = align_signals(original_array, final_array)

        # Use SNR (Signal-to-Noise Ratio) for robust quality assessment
        if len(aligned_ref) > 0 and len(aligned_test) > 0:
            noise = aligned_ref.astype(np.float64) - aligned_test.astype(np.float64)
            signal_power = np.mean(aligned_ref.astype(np.float64) ** 2)
            noise_power = np.mean(noise**2)

            if signal_power > 0 and noise_power > 0:
                snr_db = 10 * np.log10(
                    signal_power / (noise_power + 1e-20)
                )  # epsilon to avoid log(0)
                min_snr = 30  # dB (≈3% RMS error)

                assert snr_db >= min_snr, f"SNR {snr_db:.1f} dB below {min_snr} dB requirement"
            elif noise_power == 0:
                # Perfect match - signals are identical after alignment
                pass
            else:
                # Signal has no power - shouldn't happen with our test signals
                pytest.fail("Test signal has no power")
