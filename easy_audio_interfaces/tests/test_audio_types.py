import pytest

from easy_audio_interfaces import AudioChunk, AudioFormat


def test_audio_format_properties():
    """Intent: native PCM format metadata exposes the interleaved frame size."""
    audio_format = AudioFormat(rate=16_000, width=2, channels=2)

    assert audio_format.frame_size == 4


def test_audio_chunk_duration_and_format():
    """Intent: native chunks derive consistent sample counts, durations, and format data."""
    chunk = AudioChunk(
        audio=b"\x00\x00" * 160,
        rate=16_000,
        width=2,
        channels=1,
        timestamp=123,
    )

    assert chunk.samples == 160
    assert chunk.seconds == 0.01
    assert chunk.milliseconds == 10
    assert chunk.format == AudioFormat(rate=16_000, width=2, channels=1)


def test_audio_chunk_normalizes_mutable_bytes():
    """Intent: chunks own immutable bytes even when callers provide mutable buffers."""
    source = bytearray(b"\x00\x00")
    chunk = AudioChunk(audio=source, rate=16_000, width=2, channels=1)
    source[:] = b"\x01\x01"

    assert chunk.audio == b"\x00\x00"
    assert isinstance(chunk.audio, bytes)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rate": 0, "width": 2, "channels": 1}, "rate"),
        ({"rate": 16_000, "width": 0, "channels": 1}, "width"),
        ({"rate": 16_000, "width": 2, "channels": 0}, "channels"),
    ],
)
def test_audio_format_rejects_invalid_values(kwargs, message):
    """Intent: invalid PCM metadata cannot enter the audio pipeline."""
    with pytest.raises(ValueError, match=message):
        AudioFormat(**kwargs)


def test_audio_chunk_rejects_partial_sample_frame():
    """Intent: chunks reject payloads that end inside an interleaved PCM frame."""
    with pytest.raises(ValueError, match="frame size"):
        AudioChunk(audio=b"\x00\x00\x00", rate=16_000, width=2, channels=1)
