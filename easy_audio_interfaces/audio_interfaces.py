import json
import logging
import threading
import wave
from queue import Queue
from socket import AF_INET, SOCK_STREAM, socket
from types import TracebackType
from typing import Dict, Generator, Iterable, Optional, Protocol, Type, Union, cast

import numpy as np
import pyaudio
from samplerate import Resampler

from easy_audio_interfaces.types.audio import NumpyFrame
from easy_audio_interfaces.types.common import PathLike

logger = logging.getLogger(__name__)


AUDIO_FORMAT = pyaudio.paInt16


class AudioSource(Iterable, Protocol):
    """Abstract source class that can be used to read from a file or stream from microphone"""

    def read(self) -> NumpyFrame:
        ...

    def open(self):
        ...

    def close(self):
        ...

    def __enter__(self) -> "AudioSource":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        self.close()

    @property
    def sample_rate(self) -> int:
        ...


# FIXME: Add "clear_buffer" method
class InputMicStream(AudioSource):
    def __init__(
        self,
        device_index: Optional[int] = None,
        channels: int = 1,
        buffer_maxlen: Optional[int] = 1024 * 1024,
        num_frames_to_record: Optional[int] = None,
    ):
        self._p = pyaudio.PyAudio()
        self._device_index = device_index
        if self._device_index and self._device_index < 0:
            for i in range(self._p.get_device_count()):
                max_inp_channels = cast(
                    float, self._p.get_device_info_by_index(i)["maxInputChannels"]
                )
                if max_inp_channels > 0:
                    print(json.dumps(self._p.get_device_info_by_index(i), indent=4))
            self._device_index = int(input("Enter input device index: "))
        self.buffer: Queue[NumpyFrame] = Queue(maxsize=buffer_maxlen or 0)
        self._channels = channels
        self.num_frames_to_record = num_frames_to_record

    @property
    def device_info(self) -> Dict:
        if self._device_index is None:
            return cast(dict, self._p.get_default_input_device_info())
        else:
            return cast(dict, self._p.get_device_info_by_index(self._device_index))

    @property
    def sample_rate(self) -> int:
        return int(self.device_info["defaultSampleRate"])

    def open(self):
        logger.info(
            "Starting input stream using %s",
            self.device_info,
        )
        self._stream = self._p.open(
            format=AUDIO_FORMAT,
            channels=self._channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self._device_index,
            stream_callback=self._to_buffer_callback,
        )
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        logger.info("Stopping input mic stream, cleaning up.")
        if not self.is_stopped():
            self.stop()
        self.close()
        self._p.terminate()

    def __iter__(self) -> Generator[NumpyFrame, None, None]:
        while self.is_active():
            yield self.buffer.get()

    def start(self) -> "InputMicStream":
        self._stream.start_stream()
        return self

    def stop(self):
        try:
            self._stream.stop_stream()
        except OSError as e:
            logger.error(f"Error stopping stream: {e}")

    def is_stopped(self) -> bool:
        try:
            return self._stream.is_stopped()
        except OSError as e:
            logger.error(f"Error checking stream status: {e}")
            return False

    def is_active(self) -> bool:
        try:
            return self._stream.is_active()
        except OSError as e:
            logger.error(f"Error checking if stream is active: {e}")
            return False

    def _to_buffer_callback(self, in_data: Optional[bytes], frame_count, time_info, status):
        if self.num_frames_to_record is not None:
            self.num_frames_to_record -= 1
            if self.num_frames_to_record <= 0:
                self.stop()
        if in_data is not None:
            data = NumpyFrame.frombuffer(in_data)
            self.buffer.put(data)
        return None, pyaudio.paContinue

    def read(self) -> NumpyFrame:
        return self.buffer.get()

    def close(self):
        self._stream.close()


class InputFileStream(AudioSource):
    def __init__(self, file_path: PathLike, chunk_size: int = 1024):
        self.file_path = file_path
        self._chunk_size = chunk_size

    @property
    def sample_rate(self) -> int:
        return self._wf.getframerate()

    @property
    def channels(self) -> int:
        return self._wf.getnchannels()

    def open(self):
        self._wf = wave.open(str(self.file_path), "rb")
        logger.info(
            f"Started file stream: {self.file_path}, sample rate: {self.sample_rate}, channels: {self.channels}",
        )

    def close(self):
        self._wf.close()
        logger.info("Closed file stream.")

    def read(self) -> Optional[NumpyFrame]:
        if self._wf.tell() < self._wf.getnframes():
            return NumpyFrame.frombuffer(self._wf.readframes(self._chunk_size))
        elif self._wf.tell() + self._chunk_size >= self._wf.getnframes():
            logger.info("Called read() after EOF. Returning None.")

    def __iter__(self) -> Generator[NumpyFrame, None, None]:
        while self._wf.tell() + self._chunk_size < self._wf.getnframes():
            yield NumpyFrame.frombuffer(self._wf.readframes(self._chunk_size))
        else:
            logger.info(
                f"Finished reading frames. Pointer at {self._wf.tell()}. Discarding {self._wf.getnframes() - self._wf.tell()} frames."
            )


class SocketReceiver(AudioSource):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        buffer_maxlen: Optional[int] = None,
        port: int = 5000,
        host: str = "localhost",
    ):
        self.buffer: Queue[NumpyFrame] = Queue(maxsize=buffer_maxlen or 0)
        self._sample_rate = sample_rate
        self._channels = channels
        self._port = port
        self._host = host
        self.reader_thread = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def open(self):
        self._socket = socket(AF_INET, SOCK_STREAM)
        self._socket.bind((self._host, self._port))
        self._socket.listen(1)
        logger.info(
            f"Started socket receiver: {self._host}:{self._port}, sample rate: {self.sample_rate}, channels: {self.channels}",
        )
        self.start_receiving()

    def close(self):
        self._socket.close()
        logger.info("Closed socket receiver.")

    def _read(self):
        while True:
            try:
                data = self._connection.recv(1024)
                if data:
                    self.buffer.put(NumpyFrame.frombuffer(data))
                else:
                    logger.info("No more data from client.")
                    break
            except Exception as e:
                logger.error(f"Error reading from socket: {e}")
                break

    def start_receiving(self):
        self._connection, self._address = self._socket.accept()
        logger.info(f"Accepted connection from {self._address}")
        self.reader_thread = threading.Thread(target=self._read, daemon=True)
        self.reader_thread.start()

    def read(self) -> NumpyFrame:
        return self.buffer.get()

    def __iter__(self) -> Generator[NumpyFrame, None, None]:
        while True:
            yield self.read()


class AudioSink(Protocol):
    """Abstract sink class that can be used to write to a file or stream to speaker"""

    reader_thread: Optional[threading.Thread]

    def write(self, frame: NumpyFrame):
        ...

    def open(self):
        ...

    def close(self):
        ...

    def __enter__(self) -> "AudioSink":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def write_from(self, input_stream: "AudioSource"):
        ...

    @property
    def sample_rate(self) -> int:
        ...


class OutputSpeakerStream(AudioSink):
    def __init__(
        self,
        device_index: Optional[int] = None,
        channels: int = 1,
        buffer_maxlen: Optional[int] = None,
    ):
        self._p = pyaudio.PyAudio()
        self._device_index = device_index
        if self._device_index and self._device_index < 0:
            for i in range(self._p.get_device_count()):
                max_out_channels = cast(
                    float, self._p.get_device_info_by_index(i)["maxOutputChannels"]
                )
                if max_out_channels > 0:
                    print(json.dumps(self._p.get_device_info_by_index(i), indent=4))
            self._device_index = int(input("Enter output device index: "))
        self.buffer: Queue[NumpyFrame] = Queue(maxsize=buffer_maxlen or 0)
        self._channels = channels
        self.reader_thread = None

    @property
    def device_info(self) -> Dict:
        if self._device_index is None:
            return cast(dict, self._p.get_default_output_device_info())
        else:
            return cast(dict, self._p.get_device_info_by_index(self._device_index))

    @property
    def sample_rate(self) -> int:
        return int(self.device_info["defaultSampleRate"])

    def open(self):
        logger.info(
            "Starting output stream using %s",
            self.device_info,
        )
        self._stream = self._p.open(
            rate=self.sample_rate,
            format=AUDIO_FORMAT,
            output=True,
            channels=self._channels,
            output_device_index=self._device_index,
        )

    def close(self):
        self._stream.stop_stream()
        self._stream.close()
        self._p.terminate()
        logger.info("Stopped speaker stream.")

    def write(self, data: NumpyFrame):
        self._stream.write(data.tobytes())

    def write_from(self, input_stream: Union[AudioSource, "AudioProcessingBlock"]):
        def _write():
            for chunk in input_stream:
                self.write(chunk)

        self.reader_thread = threading.Thread(target=_write, daemon=True)
        self.reader_thread.start()


class OutputFileStream(AudioSink):
    def __init__(self, file_path: PathLike, sample_rate: int = 48000, channels: int = 1):
        self.file_path = str(file_path)
        self._sample_rate = sample_rate
        self.channels = channels
        self.reader_thread = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def open(self):
        self._file = wave.open(self.file_path, "wb")
        self._file.setnchannels(self.channels)
        self._file.setsampwidth(2)  # 16 bit
        self._file.setframerate(self.sample_rate)
        self._is_open = True

        self.numel = 0

    def close(self):
        logger.info("Stopping output file stream, cleanup.")
        self._is_open = False
        self._file.close()

    def is_open(self) -> bool:
        return self._is_open

    def write(self, data: NumpyFrame):
        if self.is_open():
            self.numel += data.shape[0]
            self._file.writeframes(data.tobytes())
        else:
            raise RuntimeError("File is not open.")

    def write_from(self, input_stream: Union[AudioSource, "AudioProcessingBlock"]):
        def _write():
            for chunk in input_stream:
                self.write(chunk)

        self.reader_thread = threading.Thread(target=_write, daemon=True)
        self.reader_thread.start()


class SocketStreamer(AudioSink):
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        buffer_maxlen: Optional[int] = None,
        port: int = 5000,
        host: str = "localhost",
    ):
        self.buffer: Queue[NumpyFrame] = Queue(maxsize=buffer_maxlen or 0)
        self._sample_rate = sample_rate
        self._channels = channels
        self._port = port
        self._host = host
        self.reader_thread = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def open(self):
        self._socket = socket(AF_INET, SOCK_STREAM)
        self._socket.connect((self._host, self._port))
        logger.info(
            f"Started socket streamer: {self._host}:{self._port}, sample rate: {self.sample_rate}, channels: {self.channels}",
        )

    def close(self):
        self._socket.close()
        logger.info("Closed socket streamer.")

    def write(self, data: NumpyFrame):
        self._socket.send(data.tobytes())

    def write_from(self, input_stream: Union[AudioSource, "AudioProcessingBlock"]):
        def _write():
            for chunk in input_stream:
                self.write(chunk)

        self.reader_thread = threading.Thread(target=_write, daemon=True)
        self.reader_thread.start()


class AudioProcessingBlock(Iterable, Protocol):
    """Abstract class that can be used to process audio data."""

    reader_thread: Optional[threading.Thread]

    # If the input stream is not active after init, we can assume that the previous stream has ended,
    # and use this information to yield the last chunk of data.
    input_stream_terminated: bool = False

    def open(self):
        ...

    def close(self):
        ...

    def __enter__(self) -> "AudioProcessingBlock":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    def write(self, data: NumpyFrame) -> None:
        ...

    def read(self) -> NumpyFrame:
        ...

    def write_from(self, input_stream: "AudioSource"):
        ...

    @property
    def sample_rate(self) -> int:
        ...


class CollectorBlock(AudioProcessingBlock):
    def __init__(
        self,
        sample_rate: int,
        collect_seconds: float,
        buffer_maxlen: Optional[int] = None,
    ):
        self.samples: Queue[NumpyFrame] = Queue(maxsize=buffer_maxlen or 0)
        self._sample_rate = sample_rate
        self._collect_seconds = collect_seconds
        self.reader_thread = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def open(self):
        ...

    def close(self):
        ...

    def write(self, data: NumpyFrame):
        for sample in data:
            self.samples.put(sample)

    def write_from(self, input_stream: Union[AudioSource, "AudioProcessingBlock"]):
        def _write():
            for chunk in input_stream:
                self.write(chunk)

        self.reader_thread = threading.Thread(target=_write, daemon=True)
        self.reader_thread.start()

    def read(self) -> NumpyFrame:
        samples = []
        while (
            not self.input_stream_terminated
            and len(samples) < self._collect_seconds * self._sample_rate
        ):
            samples.append(self.samples.get())
        return NumpyFrame(np.array(samples))

    def __iter__(self) -> Generator[NumpyFrame, None, None]:
        while True:
            yield self.read()


class ResamplingBlock(AudioProcessingBlock):
    class PreBuffer:
        def __init__(self, buffer_maxlen: Optional[int] = None):
            self.buffer: Queue[NumpyFrame] = Queue(maxsize=buffer_maxlen or 0)
            self.total_frames = 0

        def put(self, data: NumpyFrame):
            self.total_frames += data.shape[0]
            self.buffer.put(data)

        def get(self, chunk_size: int) -> Optional[NumpyFrame]:
            if self.total_frames >= chunk_size:
                frames = []
                while chunk_size > 0:
                    frames.append(self.buffer.get())
                    chunk_size -= frames[-1].shape[0]
                self.total_frames -= sum(f.shape[0] for f in frames)
                return NumpyFrame(np.concatenate(frames))
            else:
                return None

    def __init__(
        self,
        original_sample_rate: int = 48000,
        resample_rate: int = 16000,
        conversion_method: str = "sinc_fastest",
        chunk_size: int = 1024,
        buffer_maxlen: Optional[int] = None,
    ):
        self.buffer: Queue[NumpyFrame] = Queue(maxsize=buffer_maxlen or 0)
        self.prebuffer = self.PreBuffer(buffer_maxlen=buffer_maxlen)
        self._original_sample_rate = original_sample_rate
        self._resample_rate = resample_rate
        self._chunk_size = chunk_size
        self.resampler = Resampler(converter_type=conversion_method, channels=1)
        self.reader_thread = None

    def open(self):
        ...

    def close(self):
        ...

    @property
    def sample_rate(self) -> int:
        return self._resample_rate

    def write(self, data: NumpyFrame):
        self.prebuffer.put(data)
        chunk = self.prebuffer.get(self._chunk_size)
        if chunk is not None:
            self.buffer.put(self._resample_chunk(chunk))

    def _resample_chunk(self, chunk: NumpyFrame) -> NumpyFrame:
        return NumpyFrame(
            self.resampler.process(
                chunk, ratio=self._resample_rate / self._original_sample_rate
            ).astype(np.int16)
        )

    def write_from(self, input_stream: Union[AudioSource, "AudioProcessingBlock"]):
        def _write():
            for chunk in input_stream:
                self.write(chunk)

        self.reader_thread = threading.Thread(target=_write, daemon=True)
        self.reader_thread.start()

    def __iter__(self) -> Generator[NumpyFrame, None, None]:
        while True:
            yield self.buffer.get()

    def read(self) -> NumpyFrame:
        return self.buffer.get()
