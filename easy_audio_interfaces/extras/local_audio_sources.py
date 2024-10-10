import asyncio
import logging
from typing import Optional, Type

import numpy as np
import pyaudio

from easy_audio_interfaces.audio_interfaces import AudioSource
from easy_audio_interfaces.types.audio import NumpyFrame

logger = logging.getLogger(__name__)


class PyAudioInterface:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        print(self.list_input_devices())

    def __del__(self):
        self.p.terminate()

    def get_input_device_info(self, device_index=None):
        if device_index is None:
            device_index = self.p.get_default_input_device_info()["index"]
        return self.p.get_device_info_by_index(device_index)

    def list_input_devices(self):
        return [
            self.p.get_device_info_by_index(i)
            for i in range(self.p.get_device_count())
            if self.p.get_device_info_by_index(i)["maxInputChannels"] > 0
        ]


class InputMicStream(AudioSource):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._device_index = device_index
        self._stream = None
        self._pa_interface = PyAudioInterface()
        self._stop_event = asyncio.Event()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        device_info = self._pa_interface.get_input_device_info(self._device_index)
        logger.info(f"Opening input stream on device: {device_info['name']}")
        self._stream = self._pa_interface.p.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=self._chunk_size,
        )
        logger.info("Input stream opened successfully")

    async def close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            logger.info("Input stream closed")
        self._stop_event.set()

    async def read(self) -> NumpyFrame:
        if self._stream is None:
            raise RuntimeError("Stream is not open. Call 'open()' first.")

        data = self._stream.read(self._chunk_size)
        return NumpyFrame(np.frombuffer(data, dtype=np.int16))

    async def iter_frames(self):
        while not self._stop_event.is_set():
            frame = await self.read()
            yield frame

    async def __aenter__(self) -> "InputMicStream":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ):
        await self.close()
