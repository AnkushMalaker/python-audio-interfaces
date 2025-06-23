import asyncio
import logging
import time
from typing import Optional, Type

import pyaudio
from wyoming.audio import AudioChunk

from easy_audio_interfaces.audio_interfaces import AudioSink, AudioSource

logger = logging.getLogger(__name__)


class PyAudioInterface:
    def __init__(self):
        self.p = pyaudio.PyAudio()

    def __del__(self):
        self.p.terminate()

    def get_input_device_info(self, device_index=None):
        if device_index is None:
            device_index = self.p.get_default_input_device_info()["index"]
        return self.p.get_device_info_by_index(int(device_index))

    def list_input_devices(self):
        return [
            self.p.get_device_info_by_index(i)
            for i in range(self.p.get_device_count())
            if int(self.p.get_device_info_by_index(i)["maxInputChannels"]) > 0
        ]

    def get_output_device_info(self, device_index=None):
        if device_index is None:
            device_index = self.p.get_default_output_device_info()["index"]
        return self.p.get_device_info_by_index(int(device_index))


class InputMicStream(PyAudioInterface, AudioSource):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        PyAudioInterface.__init__(self)
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._device_index = device_index
        self._stream = None
        self._stop_event = asyncio.Event()

    @property
    def sample_width(self) -> int:
        return 2  # 16-bit audio

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        device_info = self.get_input_device_info(self._device_index)
        logger.info(f"Opening input stream on device: {device_info['name']}")
        self._stream = self.p.open(
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

    async def read(self) -> AudioChunk:
        if self._stream is None:
            raise RuntimeError("Stream is not open. Call 'open()' first.")

        data = self._stream.read(self._chunk_size)
        return AudioChunk(
            audio=data,
            rate=self._sample_rate,
            width=2,
            channels=self._channels,
            timestamp=int(time.time()),
        )

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


class OutputSpeakerStream(PyAudioInterface, AudioSink):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        device_index: Optional[int] = None,
    ):
        PyAudioInterface.__init__(self)
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._device_index = device_index
        self._stream = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    async def open(self):
        device_info = self.get_output_device_info(self._device_index)
        logger.info(f"Opening output stream on device: {device_info['name']}")
        self._stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sample_rate,
            output=True,
            output_device_index=self._device_index,
            frames_per_buffer=self._chunk_size,
        )
        logger.info("Output stream opened successfully")

    async def close(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
            logger.info("Output stream closed")

    def _validate_chunk_parameters(self, frame: AudioChunk):
        if frame.rate != self._sample_rate:
            raise ValueError(f"Sample rate mismatch: {frame.rate} != {self._sample_rate}")
        if frame.width != 2:
            raise ValueError(f"Width mismatch: {frame.width} != 2")
        if frame.channels != self._channels:
            raise ValueError(f"Channels mismatch: {frame.channels} != {self._channels}")

    def write(self, frame: AudioChunk):
        self._validate_chunk_parameters(frame)
        if self._stream is None:
            raise RuntimeError("Stream is not open. Call 'open()' first.")
        self._stream.write(frame.audio)  # type: ignore

    async def __aenter__(self) -> "OutputSpeakerStream":
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[Type[BaseException]],
    ):
        await self.close()

    async def write_from(self, audio_source: AudioSource):
        async def _write_from():
            try:
                async for frame in audio_source:
                    self.write(frame)
            except asyncio.CancelledError:
                logger.info("write_from task cancelled")

        await _write_from()


async def main():
    RUN_FOR = 10
    async with InputMicStream() as mic, OutputSpeakerStream() as speaker:
        # This is the non-blocking version.
        # write_task = asyncio.create_task(speaker.write_from(mic))
        # await asyncio.sleep(RUN_FOR)
        # write_task.cancel()
        # await write_task  # Wait for the task to be cancelled

        # Or simply:
        # await speaker.write_from(mic)

        # or even more simply:
        async for frame in mic:
            frame = frame + 5  # Do some processing
            print(frame.dBFS)
            speaker.write(frame)


if __name__ == "__main__":
    asyncio.run(main())
