import wave
from pathlib import Path
from typing import BinaryIO, Generator, Optional

from wyoming.audio import AudioChunk


class WaveFileStreamer:
    """Streaming utility for reading wave files in chunks."""

    def __init__(self, file_like: BinaryIO | str | Path):
        self._file_like = file_like
        self._wav_file: Optional[wave.Wave_read] = None
        self._rate: Optional[int] = None
        self._width: Optional[int] = None
        self._channels: Optional[int] = None
        self._frames_remaining: Optional[int] = None

    def open(self) -> "WaveFileStreamer":
        """Open the wave file for reading."""
        # Convert Path objects to strings for wave.open()
        file_arg = str(self._file_like) if isinstance(self._file_like, Path) else self._file_like
        self._wav_file = wave.open(file_arg, "rb")  # type: ignore
        self._rate = self._wav_file.getframerate()
        self._width = self._wav_file.getsampwidth()
        self._channels = self._wav_file.getnchannels()
        self._frames_remaining = self._wav_file.getnframes()
        return self

    def close(self):
        """Close the wave file."""
        if self._wav_file:
            self._wav_file.close()
            self._wav_file = None

    def __enter__(self) -> "WaveFileStreamer":
        return self.open()

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        self.close()

    @property
    def rate(self) -> int:
        if self._rate is None:
            raise RuntimeError("WaveFileStreamer not opened")
        return self._rate

    @property
    def width(self) -> int:
        if self._width is None:
            raise RuntimeError("WaveFileStreamer not opened")
        return self._width

    @property
    def channels(self) -> int:
        if self._channels is None:
            raise RuntimeError("WaveFileStreamer not opened")
        return self._channels

    @property
    def frames_remaining(self) -> int:
        if self._frames_remaining is None:
            raise RuntimeError("WaveFileStreamer not opened")
        return self._frames_remaining

    def read_chunk_by_samples(self, num_samples: int) -> AudioChunk:
        """Read a chunk of the specified number of samples."""
        if self._wav_file is None or self._frames_remaining is None:
            raise RuntimeError("WaveFileStreamer not opened")

        if self._frames_remaining <= 0:
            raise StopIteration

        frames_to_read = min(num_samples, self._frames_remaining)
        audio_data = self._wav_file.readframes(frames_to_read)
        self._frames_remaining -= frames_to_read

        return AudioChunk(
            audio=audio_data,
            rate=self.rate,
            width=self.width,
            channels=self.channels,
        )

    def read_chunk_by_duration(self, duration_ms: int) -> AudioChunk:
        """Read a chunk of the specified duration in milliseconds."""
        if self._rate is None:
            raise RuntimeError("WaveFileStreamer not opened")
        samples_per_ms = self._rate / 1000
        num_samples = int(duration_ms * samples_per_ms)
        return self.read_chunk_by_samples(num_samples)

    def iter_chunks_by_samples(self, num_samples: int) -> Generator[AudioChunk, None, None]:
        """Iterate over chunks of the specified number of samples."""
        while self._frames_remaining is not None and self._frames_remaining > 0:
            try:
                yield self.read_chunk_by_samples(num_samples)
            except StopIteration:
                break

    def iter_chunks_by_duration(self, duration_ms: int) -> Generator[AudioChunk, None, None]:
        """Iterate over chunks of the specified duration in milliseconds."""
        while self._frames_remaining is not None and self._frames_remaining > 0:
            try:
                yield self.read_chunk_by_duration(duration_ms)
            except StopIteration:
                break
