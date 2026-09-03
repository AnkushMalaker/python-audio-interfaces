import io
import wave

import pytest

from easy_audio_interfaces.streaming import WaveFileStreamer


def _wav_bytes(*, rate: int = 800) -> io.BytesIO:
    file_like = io.BytesIO()
    with wave.open(file_like, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(rate)
        wav_file.writeframes(b"\x00\x00" * 8)
    file_like.seek(0)
    return file_like


@pytest.mark.parametrize("num_samples", [0, -1])
def test_wave_streamer_rejects_non_positive_sample_chunks(num_samples):
    """Intent: invalid sample chunk sizes cannot leave iteration stuck on the same frame."""
    with WaveFileStreamer(_wav_bytes()) as streamer:
        with pytest.raises(ValueError, match="num_samples must be greater than zero"):
            streamer.read_chunk_by_samples(num_samples)


@pytest.mark.parametrize("duration_ms", [0, -1])
def test_wave_streamer_rejects_non_positive_duration_chunks(duration_ms):
    """Intent: invalid duration chunk sizes fail before a WAV iterator can stall."""
    with WaveFileStreamer(_wav_bytes()) as streamer:
        with pytest.raises(ValueError, match="duration_ms must be greater than zero"):
            streamer.read_chunk_by_duration(duration_ms)


def test_wave_streamer_rejects_duration_below_one_sample():
    """Intent: positive millisecond chunks must still advance at low sample rates."""
    with WaveFileStreamer(_wav_bytes(rate=800)) as streamer:
        with pytest.raises(ValueError, match="too small to contain one sample"):
            streamer.read_chunk_by_duration(1)
