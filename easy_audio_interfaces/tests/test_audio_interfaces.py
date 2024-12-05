# test_audio_interfaces.py

import asyncio
from typing import AsyncGenerator, AsyncIterable, cast

import pytest
from pydub import AudioSegment, generators

from easy_audio_interfaces import LocalFileSink, RechunkingBlock, ResamplingBlock

SINE_FREQUENCY = 440
SINE_SAMPLE_RATE = 44100


def create_sine_wave_segment(duration_ms: int) -> AudioSegment:
    sine_wave = generators.Sine(freq=SINE_FREQUENCY, sample_rate=int(SINE_SAMPLE_RATE))
    return sine_wave.to_audio_segment(duration=duration_ms)


async def async_generator(wav_segment: AudioSegment) -> AsyncGenerator[AudioSegment, None]:
    # Split the 10-second wav_segment into 1-second chunks
    chunk_duration_ms = 1000
    for i in range(0, len(wav_segment), chunk_duration_ms):
        chunk = cast(AudioSegment, wav_segment[i : i + chunk_duration_ms])
        await asyncio.sleep(0.0)  # simulate async
        yield chunk


@pytest.mark.asyncio
async def test_resampling_block():
    # Create a sample AudioSegment
    original_sample_rate = 16000
    new_sample_rate = 22050
    duration_ms = 10000  # 10 seconds

    wav_segment = create_sine_wave_segment(duration_ms).set_frame_rate(original_sample_rate)

    async with LocalFileSink(
        "test_resampling_block_input.wav", sample_rate=original_sample_rate, channels=1
    ) as file_sink:
        async for chunk in async_generator(wav_segment):
            await file_sink.write(chunk)

    # Initialize ResamplingBlock
    resampler = ResamplingBlock(resample_rate=new_sample_rate)

    # Process the frame
    wav_segment = create_sine_wave_segment(duration_ms).set_frame_rate(original_sample_rate)
    output_segments = []
    async with LocalFileSink(
        "test_resampling_block_output.wav", sample_rate=new_sample_rate, channels=1
    ) as file_sink:
        async for output_segment in resampler.process(async_generator(wav_segment)):
            output_segments.append(output_segment)
            await file_sink.write(output_segment)

    # Check that the output segment has the new sample rate
    assert len(output_segments) == 10
    assert output_segments[0].frame_rate == new_sample_rate


@pytest.mark.asyncio
async def test_rechunking_block_in_ms():
    # Create a sample AudioSegment of 5.5 seconds duration
    duration_ms = 10000  # 10 seconds
    wav_segment = create_sine_wave_segment(duration_ms)

    # Initialize RechunkingBlock with chunk size 1 second (1000 ms)
    chunk_size_ms = 500
    rechunker = RechunkingBlock(chunk_size_ms=chunk_size_ms)

    # Process the frame
    output_segments = []
    async for output_segment in rechunker.process(async_generator(wav_segment)):
        output_segments.append(output_segment)

    # Check that we have 10 chunks
    num_expected_chunks = 10000 // chunk_size_ms

    # should be 20 chunks of 500 ms
    assert len(output_segments) == num_expected_chunks
    for i, segment in enumerate(output_segments):
        assert abs(len(segment) - chunk_size_ms) < 50  # Allow small discrepancy


@pytest.mark.asyncio
async def test_rechunking_block_in_samples():
    # Create a sample AudioSegment of 5.5 seconds duration
    duration_ms = 10000  # 10 seconds
    wav_segment = create_sine_wave_segment(duration_ms)

    # Initialize RechunkingBlock with chunk size 1 second (1000 ms)
    chunk_size_samples = 512
    rechunker = RechunkingBlock(chunk_size_samples=chunk_size_samples)

    # Process the frame
    output_segments = []
    async for output_segment in rechunker.process(async_generator(wav_segment)):
        output_segments.append(output_segment)

    # Check that we have 10 chunks
    num_expected_chunks = (duration_ms // 1000) * SINE_SAMPLE_RATE // chunk_size_samples

    # should be 20 chunks of 500 ms
    assert (
        abs(len(output_segments) - num_expected_chunks) <= 1  # Last chunk may be fractional
    ), f"Expected {num_expected_chunks} chunks, got {len(output_segments)}"
    for i, segment in enumerate(output_segments[:-2]):  # Last chunk may be lesser
        assert (
            segment.frame_count() == chunk_size_samples
        ), f"Segment {i} has {segment.frame_count()} samples, total chunks: {len(output_segments)}"


@pytest.mark.asyncio
async def test_block_chaining():
    # Create a sample AudioSegment
    new_sample_rate = 22050
    duration_ms = 10000  # 10 seconds

    wav_segment = create_sine_wave_segment(duration_ms)

    # Initialize ResamplingBlock and RechunkingBlock
    resampler = ResamplingBlock(resample_rate=new_sample_rate)
    chunk_size_ms = 512
    rechunker = RechunkingBlock(chunk_size_ms=chunk_size_ms)

    # Chain the blocks
    async def _process_stream():
        resampled_stream = resampler.process(async_generator(wav_segment))
        async for chunk in rechunker.process(resampled_stream):
            yield chunk

    # Collect output segments
    output_segments = []
    async for output_segment in _process_stream():
        output_segments.append(output_segment)

    # Check that we have 10 chunks
    num_expected_chunks = duration_ms // chunk_size_ms
    assert abs(len(output_segments) - num_expected_chunks) <= 1
    for i, segment in enumerate(output_segments[:-2]):  # Last chunk may be lesser
        assert abs(len(segment) - chunk_size_ms) < 50  # Allow small discrepancy
        # Check that the sample rate is the new sample rate
        assert segment.frame_rate == new_sample_rate
