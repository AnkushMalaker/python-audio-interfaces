"""Async adapters for local microphones and speakers backed by PyAudio."""

from __future__ import annotations

import asyncio
import logging
import time
from importlib import import_module
from typing import Any, AsyncIterator

from easy_audio_interfaces.audio_interfaces import AudioSink, AudioSource
from easy_audio_interfaces.types.audio import AudioChunk
from easy_audio_interfaces.types.common import AudioStream

logger = logging.getLogger(__name__)


def _load_pyaudio() -> Any:
    """Load the optional PyAudio dependency only when local audio is requested."""

    try:
        return import_module("pyaudio")
    except ModuleNotFoundError as exc:
        if exc.name != "pyaudio":
            raise
        raise ModuleNotFoundError(
            "Local audio requires PyAudio; install easy-audio-interfaces[local-audio]."
        ) from exc


def _validate_configuration(sample_rate: int, channels: int, chunk_size: int) -> None:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if channels <= 0:
        raise ValueError("channels must be greater than zero")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")


def _stop_and_close(stream: Any) -> None:
    """Close a PortAudio stream even when stopping it reports an error."""

    try:
        stream.stop_stream()
    finally:
        stream.close()


class PyAudioInterface:
    """Own one PyAudio backend for the lifetime of a local audio stream."""

    def __init__(self, *, _pyaudio_module: Any | None = None) -> None:
        self._pyaudio = _load_pyaudio() if _pyaudio_module is None else _pyaudio_module
        self.p = self._pyaudio.PyAudio()
        self._terminated = False

    def __del__(self) -> None:
        # ``__init__`` may have failed before all attributes were installed.
        try:
            self._terminate()
        except Exception:
            pass

    def _ensure_active(self) -> None:
        if self._terminated:
            raise RuntimeError("Audio interface is closed and cannot be reopened.")

    def _terminate(self) -> None:
        if not self._terminated:
            self.p.terminate()
            self._terminated = True

    def get_input_device_info(self, device_index: int | None = None) -> dict[str, Any]:
        self._ensure_active()
        if device_index is None:
            device_index = int(self.p.get_default_input_device_info()["index"])
        return self.p.get_device_info_by_index(int(device_index))

    def list_input_devices(self) -> list[dict[str, Any]]:
        self._ensure_active()
        devices = [
            self.p.get_device_info_by_index(index) for index in range(self.p.get_device_count())
        ]
        return [device for device in devices if int(device["maxInputChannels"]) > 0]

    def get_output_device_info(self, device_index: int | None = None) -> dict[str, Any]:
        self._ensure_active()
        if device_index is None:
            device_index = int(self.p.get_default_output_device_info()["index"])
        return self.p.get_device_info_by_index(int(device_index))

    def list_output_devices(self) -> list[dict[str, Any]]:
        self._ensure_active()
        devices = [
            self.p.get_device_info_by_index(index) for index in range(self.p.get_device_count())
        ]
        return [device for device in devices if int(device["maxOutputChannels"]) > 0]


class InputMicStream(PyAudioInterface, AudioSource):
    """Read 16-bit PCM chunks from a local input device."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: int | None = None,
        *,
        _pyaudio_module: Any | None = None,
    ) -> None:
        _validate_configuration(sample_rate, channels, chunk_size)
        PyAudioInterface.__init__(self, _pyaudio_module=_pyaudio_module)
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._device_index = device_index
        self._stream: Any | None = None
        self._stop_event = asyncio.Event()

    @property
    def sample_width(self) -> int:
        return 2

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def _open_input_stream(self) -> Any:
        device_info = self.get_input_device_info(self._device_index)
        logger.info("Opening input stream on device: %s", device_info["name"])
        return self.p.open(
            format=self._pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self._chunk_size,
        )

    async def open(self) -> None:
        self._ensure_active()
        if self._stream is not None:
            raise RuntimeError("Input stream is already open.")
        self._stop_event.clear()
        self._stream = await asyncio.to_thread(self._open_input_stream)
        logger.info("Input stream opened successfully")

    async def close(self) -> None:
        self._stop_event.set()
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                await asyncio.to_thread(_stop_and_close, stream)
                logger.info("Input stream closed")
        finally:
            await asyncio.to_thread(self._terminate)

    async def read(self) -> AudioChunk:
        stream = self._stream
        if stream is None:
            raise RuntimeError("Stream is not open. Call 'open()' first.")

        data = await asyncio.to_thread(stream.read, self._chunk_size)
        return AudioChunk(
            audio=data,
            rate=self._sample_rate,
            width=self.sample_width,
            channels=self._channels,
            timestamp=int(time.time()),
        )

    async def iter_frames(self) -> AsyncIterator[AudioChunk]:
        while not self._stop_event.is_set():
            try:
                frame = await self.read()
            except RuntimeError:
                if self._stop_event.is_set():
                    break
                raise
            yield frame

    async def __aenter__(self) -> InputMicStream:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[BaseException] | None,
    ) -> None:
        await self.close()


class OutputSpeakerStream(PyAudioInterface, AudioSink):
    """Write 16-bit PCM chunks to a local output device."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: int | None = None,
        *,
        _pyaudio_module: Any | None = None,
    ) -> None:
        _validate_configuration(sample_rate, channels, chunk_size)
        PyAudioInterface.__init__(self, _pyaudio_module=_pyaudio_module)
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._device_index = device_index
        self._stream: Any | None = None

    @property
    def sample_width(self) -> int:
        return 2

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def _open_output_stream(self) -> Any:
        device_info = self.get_output_device_info(self._device_index)
        logger.info("Opening output stream on device: %s", device_info["name"])
        return self.p.open(
            format=self._pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            output=True,
            output_device_index=self._device_index,
            frames_per_buffer=self._chunk_size,
        )

    async def open(self) -> None:
        self._ensure_active()
        if self._stream is not None:
            raise RuntimeError("Output stream is already open.")
        self._stream = await asyncio.to_thread(self._open_output_stream)
        logger.info("Output stream opened successfully")

    async def close(self) -> None:
        stream, self._stream = self._stream, None
        try:
            if stream is not None:
                await asyncio.to_thread(_stop_and_close, stream)
                logger.info("Output stream closed")
        finally:
            await asyncio.to_thread(self._terminate)

    def _validate_chunk_parameters(self, frame: AudioChunk) -> None:
        if frame.rate != self._sample_rate:
            raise ValueError(f"Sample rate mismatch: {frame.rate} != {self._sample_rate}")
        if frame.width != self.sample_width:
            raise ValueError(f"Width mismatch: {frame.width} != {self.sample_width}")
        if frame.channels != self._channels:
            raise ValueError(f"Channels mismatch: {frame.channels} != {self._channels}")

    async def write(self, frame: AudioChunk) -> None:
        self._validate_chunk_parameters(frame)
        stream = self._stream
        if stream is None:
            raise RuntimeError("Stream is not open. Call 'open()' first.")
        await asyncio.to_thread(stream.write, frame.audio)

    async def __aenter__(self) -> OutputSpeakerStream:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: type[BaseException] | None,
    ) -> None:
        await self.close()

    async def write_from(self, audio_source: AudioStream) -> None:
        async for frame in audio_source:
            await self.write(frame)
