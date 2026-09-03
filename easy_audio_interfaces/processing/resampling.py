"""Stateful sample-rate conversion for raw interleaved PCM chunks."""

from typing import Final

import numpy as np
import soxr
from numpy.typing import NDArray

from easy_audio_interfaces.types.audio import AudioChunk, AudioFormat

SUPPORTED_QUALITIES: Final = frozenset({"QQ", "LQ", "MQ", "HQ", "VHQ"})
SUPPORTED_WIDTHS: Final = frozenset({2, 4})


def _pcm_to_float32(chunk: AudioChunk) -> NDArray[np.float32]:
    """Decode signed little-endian PCM into SoXR's normalized float format."""

    if chunk.width == 2:
        samples = np.frombuffer(chunk.audio, dtype="<i2")
        scale = 1 << 15
    elif chunk.width == 4:
        samples = np.frombuffer(chunk.audio, dtype="<i4")
        scale = 1 << 31
    else:
        raise NotImplementedError("sample-rate conversion supports 16-bit and 32-bit PCM")

    normalized = (samples.astype(np.float32) / np.float32(scale)).astype(np.float32, copy=False)
    if chunk.channels > 1:
        return normalized.reshape(-1, chunk.channels)
    return normalized


def _float32_to_pcm(samples: NDArray[np.float32], width: int) -> bytes:
    """Encode normalized float samples as signed little-endian PCM."""

    if width == 2:
        minimum, maximum, scale = -(1 << 15), (1 << 15) - 1, 1 << 15
    elif width == 4:
        minimum, maximum, scale = -(1 << 31), (1 << 31) - 1, 1 << 31
    else:
        raise NotImplementedError("sample-rate conversion supports 16-bit and 32-bit PCM")

    encoded = np.rint(samples.astype(np.float64) * scale)
    clipped = np.clip(encoded, minimum, maximum)
    if width == 2:
        return clipped.astype("<i2").tobytes()
    return clipped.astype("<i4").tobytes()


class SampleRateConverter:
    """Incrementally convert a single PCM stream to another sample rate.

    The converter owns SoXR's filter state. Call :meth:`push` for every source
    chunk, then call :meth:`finish` exactly at the logical end of the stream.
    ``finish`` is idempotent, while pushing more data after it is an error until
    :meth:`reset` starts a new stream.

    This class deliberately changes only the sample rate. Channel mixing, sample
    width conversion, packet sizing, clocks, and transport semantics belong in
    separate layers. Input chunks must therefore keep one format for the entire
    stream. Signed 16-bit and 32-bit little-endian PCM are supported.
    """

    def __init__(self, output_rate: int, *, quality: str = "VHQ") -> None:
        if output_rate <= 0:
            raise ValueError("output_rate must be greater than zero")
        if quality not in SUPPORTED_QUALITIES:
            choices = ", ".join(sorted(SUPPORTED_QUALITIES))
            raise ValueError(f"quality must be one of {choices}, got: {quality}")

        self._output_rate = output_rate
        self._quality = quality
        self._input_format: AudioFormat | None = None
        self._stream: soxr.ResampleStream | None = None
        self._finished = False

    @property
    def input_format(self) -> AudioFormat | None:
        """Format inferred from the first pushed chunk, if any."""

        return self._input_format

    @property
    def output_rate(self) -> int:
        """Configured output sample rate in Hz."""

        return self._output_rate

    @property
    def output_format(self) -> AudioFormat | None:
        """Output format, available after the first pushed chunk."""

        if self._input_format is None:
            return None
        return AudioFormat(
            rate=self._output_rate,
            width=self._input_format.width,
            channels=self._input_format.channels,
        )

    @property
    def finished(self) -> bool:
        """Whether the current stream has been finalized."""

        return self._finished

    def _start(self, audio_format: AudioFormat) -> None:
        if audio_format.width not in SUPPORTED_WIDTHS:
            raise NotImplementedError("sample-rate conversion supports 16-bit and 32-bit PCM")

        self._input_format = audio_format
        if audio_format.rate != self._output_rate:
            self._stream = soxr.ResampleStream(
                audio_format.rate,
                self._output_rate,
                audio_format.channels,
                dtype="float32",
                quality=self._quality,
            )

    def _make_output(self, samples: NDArray[np.float32]) -> AudioChunk | None:
        if samples.size == 0:
            return None

        assert self._input_format is not None
        return AudioChunk(
            audio=_float32_to_pcm(samples, self._input_format.width),
            rate=self._output_rate,
            width=self._input_format.width,
            channels=self._input_format.channels,
        )

    def push(self, chunk: AudioChunk) -> AudioChunk | None:
        """Consume one chunk and return the currently available output, if any."""

        if self._finished:
            raise RuntimeError("cannot push after finish(); call reset() to start a new stream")

        if self._input_format is None:
            self._start(chunk.format)
        elif chunk.format != self._input_format:
            raise ValueError(
                f"input format changed from {self._input_format} to {chunk.format}; "
                "reset before starting a new stream"
            )

        if chunk.rate == self._output_rate:
            return chunk
        if not chunk.audio:
            return None

        assert self._stream is not None
        output = self._stream.resample_chunk(_pcm_to_float32(chunk), last=False)
        return self._make_output(output)

    def finish(self) -> AudioChunk | None:
        """Finalize the stream and return SoXR's remaining buffered output."""

        if self._finished:
            return None
        self._finished = True

        if self._input_format is None or self._stream is None:
            return None

        if self._input_format.channels == 1:
            empty = np.empty(0, dtype=np.float32)
        else:
            empty = np.empty((0, self._input_format.channels), dtype=np.float32)
        return self._make_output(self._stream.resample_chunk(empty, last=True))

    def reset(self) -> None:
        """Discard current state so this instance can convert a new stream."""

        self._input_format = None
        self._stream = None
        self._finished = False


__all__ = ["SampleRateConverter"]
