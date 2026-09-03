import numpy as np
import pytest

from easy_audio_interfaces import AudioChunk, SampleRateConverter


def sine_chunk(
    *,
    rate: int = 16_000,
    seconds: float = 1.0,
    channels: int = 1,
    width: int = 2,
) -> AudioChunk:
    """Build deterministic PCM whose packetization can vary without changing input."""

    frame_count = int(rate * seconds)
    wave = np.sin(2 * np.pi * 440 * np.arange(frame_count) / rate)
    if width == 2:
        samples = np.rint(wave * 20_000).astype("<i2")
    elif width == 4:
        samples = np.rint(wave * 1_000_000_000).astype("<i4")
    else:
        samples = np.zeros(frame_count, dtype=np.uint8)
    if channels > 1:
        samples = np.repeat(samples[:, np.newaxis], channels, axis=1)
    return AudioChunk(audio=samples.tobytes(), rate=rate, width=width, channels=channels)


def split_chunk(chunk: AudioChunk, frames_per_chunk: int) -> list[AudioChunk]:
    """Split one chunk without changing any PCM frames."""

    bytes_per_chunk = frames_per_chunk * chunk.frame_size
    return [
        AudioChunk(
            audio=chunk.audio[offset : offset + bytes_per_chunk],
            rate=chunk.rate,
            width=chunk.width,
            channels=chunk.channels,
        )
        for offset in range(0, len(chunk.audio), bytes_per_chunk)
    ]


def convert(converter: SampleRateConverter, chunks: list[AudioChunk]) -> bytes:
    """Collect non-empty incremental output and the explicit final flush."""

    output = bytearray()
    for chunk in chunks:
        converted = converter.push(chunk)
        if converted is not None:
            output.extend(converted.audio)
    final = converter.finish()
    if final is not None:
        output.extend(final.audio)
    return bytes(output)


@pytest.mark.parametrize(
    ("input_rate", "output_rate", "frames_per_chunk"),
    [(16_000, 24_000, 320), (44_100, 48_000, 137), (48_000, 16_000, 257)],
)
def test_packet_boundaries_do_not_change_resampled_audio(
    input_rate: int, output_rate: int, frames_per_chunk: int
):
    """Intent: network packetization must not restart the resampling filter."""

    source = sine_chunk(rate=input_rate, seconds=2)
    contiguous = convert(SampleRateConverter(output_rate), [source])
    packetized = convert(SampleRateConverter(output_rate), split_chunk(source, frames_per_chunk))

    assert packetized == contiguous
    assert len(packetized) // 2 == round(source.samples * output_rate / input_rate)


def test_finish_is_explicit_and_idempotent():
    """Intent: callers can drain filter latency once at a known stream boundary."""

    converter = SampleRateConverter(24_000)
    source = sine_chunk(seconds=0.02)

    initial = converter.push(source)
    final = converter.finish()

    assert converter.finished
    assert converter.finish() is None
    assert sum(chunk.samples for chunk in (initial, final) if chunk is not None) == 480


def test_push_after_finish_requires_an_explicit_reset():
    """Intent: audio from separate logical streams must never share filter history."""

    converter = SampleRateConverter(24_000)
    source = sine_chunk(seconds=0.02)
    converter.push(source)
    converter.finish()

    with pytest.raises(RuntimeError, match="cannot push after finish"):
        converter.push(source)

    replacement = sine_chunk(rate=44_100, seconds=0.02, channels=2, width=4)
    converter.reset()
    converter.push(replacement)
    assert converter.finish() is not None
    assert converter.input_format == replacement.format


@pytest.mark.parametrize(
    "changed",
    [
        sine_chunk(rate=8_000, seconds=0.02),
        sine_chunk(seconds=0.02, channels=2),
        sine_chunk(seconds=0.02, width=4),
    ],
)
def test_midstream_input_format_changes_are_rejected(changed: AudioChunk):
    """Intent: one converter state cannot silently combine incompatible PCM streams."""

    converter = SampleRateConverter(24_000)
    converter.push(sine_chunk(seconds=0.02))

    with pytest.raises(ValueError, match="input format changed"):
        converter.push(changed)


def test_equal_rates_are_a_lossless_passthrough():
    """Intent: requesting the source rate must not quantize or buffer the PCM."""

    source = sine_chunk(rate=16_000, seconds=0.02)
    converter = SampleRateConverter(16_000)

    assert converter.push(source) is source
    assert converter.finish() is None


def test_rate_conversion_does_not_invent_transport_timestamps():
    """Intent: application clocks remain outside the PCM conversion layer."""

    source = sine_chunk(seconds=0.1)
    source = AudioChunk(
        audio=source.audio,
        rate=source.rate,
        width=source.width,
        channels=source.channels,
        timestamp=123_456,
    )
    converter = SampleRateConverter(24_000)

    outputs = [converter.push(source), converter.finish()]

    assert all(chunk.timestamp is None for chunk in outputs if chunk is not None)


def test_stereo_frames_remain_interleaved_and_aligned():
    """Intent: rate conversion changes frame count without changing channel topology."""

    source = sine_chunk(rate=44_100, seconds=0.1, channels=2)
    converter = SampleRateConverter(48_000)
    output = convert(converter, split_chunk(source, 441))

    assert converter.output_format is not None
    assert converter.output_format.rate == 48_000
    assert converter.output_format.channels == 2
    assert len(output) % converter.output_format.frame_size == 0
    assert len(output) // converter.output_format.frame_size == 4_800


@pytest.mark.parametrize("output_rate", [0, -1])
def test_non_positive_output_rates_are_rejected(output_rate: int):
    """Intent: invalid clocks fail before any stream state is created."""

    with pytest.raises(ValueError, match="output_rate must be greater than zero"):
        SampleRateConverter(output_rate)


def test_unsupported_pcm_width_fails_before_conversion():
    """Intent: ambiguous 8-bit PCM signedness must not be guessed by the converter."""

    converter = SampleRateConverter(24_000)

    with pytest.raises(NotImplementedError, match="16-bit and 32-bit"):
        converter.push(sine_chunk(seconds=0.02, width=1))
