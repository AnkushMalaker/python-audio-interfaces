import pytest
from wyoming.audio import AudioChunk

from easy_audio_interfaces import LocalFileSink, RechunkingBlock

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
