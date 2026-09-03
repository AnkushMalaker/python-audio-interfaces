"""Native types for interleaved PCM audio."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Description of interleaved PCM samples.

    ``rate`` is measured in Hz, ``width`` is bytes per channel sample, and
    ``channels`` is the number of interleaved channels.
    """

    rate: int
    width: int
    channels: int

    def __post_init__(self) -> None:
        if self.rate <= 0:
            raise ValueError("rate must be greater than zero")
        if self.width not in {1, 2, 3, 4}:
            raise ValueError("width must be between 1 and 4 bytes")
        if self.channels <= 0:
            raise ValueError("channels must be greater than zero")

    @property
    def frame_size(self) -> int:
        """Number of bytes in one interleaved sample frame."""

        return self.width * self.channels


@dataclass(frozen=True, slots=True)
class AudioChunk(AudioFormat):
    """A validated chunk of raw, interleaved PCM audio."""

    audio: bytes
    timestamp: int | None = None

    def __post_init__(self) -> None:
        super(AudioChunk, self).__post_init__()

        if not isinstance(self.audio, (bytes, bytearray, memoryview)):
            raise TypeError("audio must be bytes-like")
        if not isinstance(self.audio, bytes):
            object.__setattr__(self, "audio", bytes(self.audio))

        if len(self.audio) % self.frame_size:
            raise ValueError(
                f"audio length ({len(self.audio)} bytes) must be divisible by "
                f"the frame size ({self.frame_size} bytes)"
            )

    @property
    def format(self) -> AudioFormat:
        """Return the chunk's PCM format independently of its payload."""

        return AudioFormat(rate=self.rate, width=self.width, channels=self.channels)

    @property
    def samples(self) -> int:
        """Number of sample frames in the chunk."""

        return len(self.audio) // self.frame_size

    @property
    def seconds(self) -> float:
        """Chunk duration in seconds."""

        return self.samples / self.rate

    @property
    def milliseconds(self) -> int:
        """Chunk duration in whole milliseconds."""

        return int(self.seconds * 1_000)


__all__ = ["AudioChunk", "AudioFormat"]
