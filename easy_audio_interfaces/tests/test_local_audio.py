import asyncio
import threading
from collections import defaultdict
from typing import Any, AsyncIterator

import pytest

from easy_audio_interfaces.extras import local_audio
from easy_audio_interfaces.extras.local_audio import (
    InputMicStream,
    OutputSpeakerStream,
    PyAudioInterface,
)
from easy_audio_interfaces.types.audio import AudioChunk


class FakePortAudioStream:
    def __init__(self, read_payload: bytes = b"\x01\x00\x02\x00") -> None:
        self.read_payload = read_payload
        self.read_sizes: list[int] = []
        self.writes: list[bytes] = []
        self.stop_calls = 0
        self.close_calls = 0
        self.raise_on_stop = False
        self.thread_ids: dict[str, list[int]] = defaultdict(list)

    def _record_thread(self, operation: str) -> None:
        self.thread_ids[operation].append(threading.get_ident())

    def read(self, size: int) -> bytes:
        self._record_thread("read")
        self.read_sizes.append(size)
        return self.read_payload

    def write(self, payload: bytes) -> None:
        self._record_thread("write")
        self.writes.append(payload)

    def stop_stream(self) -> None:
        self._record_thread("stop")
        self.stop_calls += 1
        if self.raise_on_stop:
            raise OSError("stop failed")

    def close(self) -> None:
        self._record_thread("close")
        self.close_calls += 1


class FakePyAudio:
    def __init__(self, stream: FakePortAudioStream | None = None) -> None:
        self.stream = stream or FakePortAudioStream()
        self.devices = [
            {"index": 0, "name": "microphone", "maxInputChannels": 1, "maxOutputChannels": 0},
            {"index": 1, "name": "speaker", "maxInputChannels": 0, "maxOutputChannels": 2},
            {"index": 2, "name": "duplex", "maxInputChannels": 1, "maxOutputChannels": 2},
        ]
        self.open_calls: list[dict[str, Any]] = []
        self.device_lookups: list[int] = []
        self.terminate_calls = 0
        self.thread_ids: dict[str, list[int]] = defaultdict(list)

    def get_default_input_device_info(self) -> dict[str, Any]:
        return self.devices[0]

    def get_default_output_device_info(self) -> dict[str, Any]:
        return self.devices[1]

    def get_device_info_by_index(self, index: int) -> dict[str, Any]:
        self.device_lookups.append(index)
        return self.devices[index]

    def get_device_count(self) -> int:
        return len(self.devices)

    def open(self, **kwargs: Any) -> FakePortAudioStream:
        self.thread_ids["open"].append(threading.get_ident())
        self.open_calls.append(kwargs)
        return self.stream

    def terminate(self) -> None:
        self.thread_ids["terminate"].append(threading.get_ident())
        self.terminate_calls += 1


class FakePyAudioModule:
    paInt16 = 8

    def __init__(self, backend: FakePyAudio) -> None:
        self.backend = backend

    def PyAudio(self) -> FakePyAudio:
        return self.backend


def fake_module(
    stream: FakePortAudioStream | None = None,
) -> tuple[FakePyAudioModule, FakePyAudio]:
    backend = FakePyAudio(stream)
    return FakePyAudioModule(backend), backend


def test_local_audio_reports_how_to_install_optional_dependency(monkeypatch) -> None:
    """Intent: importing the package stays cheap, while using local audio gives actionable setup."""

    def unavailable_dependency(name: str) -> Any:
        assert name == "pyaudio"
        raise ModuleNotFoundError("No module named 'pyaudio'", name="pyaudio")

    monkeypatch.setattr(local_audio, "import_module", unavailable_dependency)

    with pytest.raises(ModuleNotFoundError, match=r"\[local-audio\]"):
        PyAudioInterface()


@pytest.mark.parametrize("stream_type", [InputMicStream, OutputSpeakerStream])
@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sample_rate": 0}, "sample_rate"),
        ({"channels": 0}, "channels"),
        ({"chunk_size": 0}, "chunk_size"),
    ],
)
def test_invalid_stream_configuration_fails_before_opening_backend(
    stream_type, kwargs: dict[str, int], message: str
) -> None:
    """Intent: impossible local PCM configurations fail before allocating PortAudio."""

    module, backend = fake_module()

    with pytest.raises(ValueError, match=message):
        stream_type(_pyaudio_module=module, **kwargs)

    assert backend.terminate_calls == 0


def test_device_discovery_selects_defaults_and_filters_capabilities() -> None:
    """Intent: callers see only devices capable of the direction they plan to use."""

    module, backend = fake_module()
    interface = PyAudioInterface(_pyaudio_module=module)

    assert interface.get_input_device_info()["name"] == "microphone"
    assert interface.get_output_device_info()["name"] == "speaker"
    assert [device["name"] for device in interface.list_input_devices()] == [
        "microphone",
        "duplex",
    ]
    assert [device["name"] for device in interface.list_output_devices()] == [
        "speaker",
        "duplex",
    ]
    assert backend.device_lookups == [0, 1, 0, 1, 2, 0, 1, 2]
    interface._terminate()


@pytest.mark.asyncio
async def test_microphone_context_reads_pcm_and_releases_every_resource(monkeypatch) -> None:
    """Intent: microphone I/O runs off-loop and context exit closes stream and backend."""

    stream = FakePortAudioStream(b"\x01\x00\x02\x00")
    module, backend = fake_module(stream)
    event_loop_thread = threading.get_ident()
    monkeypatch.setattr(local_audio.time, "time", lambda: 1234.9)
    microphone = InputMicStream(
        sample_rate=8000,
        channels=1,
        chunk_size=2,
        device_index=2,
        _pyaudio_module=module,
    )

    async with microphone:
        frame = await microphone.read()

    assert frame == AudioChunk(
        audio=b"\x01\x00\x02\x00", rate=8000, width=2, channels=1, timestamp=1234
    )
    assert microphone.sample_rate == 8000
    assert microphone.sample_width == 2
    assert microphone.channels == 1
    assert stream.read_sizes == [2]
    assert backend.open_calls == [
        {
            "format": 8,
            "channels": 1,
            "rate": 8000,
            "input": True,
            "input_device_index": 2,
            "frames_per_buffer": 2,
        }
    ]
    assert stream.stop_calls == stream.close_calls == backend.terminate_calls == 1
    assert backend.thread_ids["open"][0] != event_loop_thread
    assert backend.thread_ids["terminate"][0] != event_loop_thread
    assert stream.thread_ids["read"][0] != event_loop_thread
    assert stream.thread_ids["stop"][0] != event_loop_thread
    assert stream.thread_ids["close"][0] != event_loop_thread


@pytest.mark.asyncio
async def test_microphone_iteration_stops_cleanly_when_closed() -> None:
    """Intent: closing a microphone terminates async iteration without one more device read."""

    stream = FakePortAudioStream()
    module, _ = fake_module(stream)
    microphone = InputMicStream(chunk_size=2, _pyaudio_module=module)
    await microphone.open()
    frames = microphone.iter_frames()

    assert (await anext(frames)).audio == stream.read_payload
    await microphone.close()

    with pytest.raises(StopAsyncIteration):
        await anext(frames)
    assert stream.read_sizes == [2]


@pytest.mark.asyncio
async def test_local_stream_state_rejects_invalid_lifecycle_operations() -> None:
    """Intent: closed or duplicate stream operations fail instead of using stale handles."""

    input_module, _ = fake_module()
    output_module, _ = fake_module()
    microphone = InputMicStream(_pyaudio_module=input_module)
    speaker = OutputSpeakerStream(_pyaudio_module=output_module)
    matching_frame = AudioChunk(audio=b"\x00\x00", rate=16000, width=2, channels=1)

    with pytest.raises(RuntimeError, match="not open"):
        await microphone.read()
    with pytest.raises(RuntimeError, match="not open"):
        await speaker.write(matching_frame)

    await microphone.open()
    await speaker.open()
    with pytest.raises(RuntimeError, match="already open"):
        await microphone.open()
    with pytest.raises(RuntimeError, match="already open"):
        await speaker.open()

    await microphone.close()
    await speaker.close()
    with pytest.raises(RuntimeError, match="cannot be reopened"):
        await microphone.open()
    with pytest.raises(RuntimeError, match="cannot be reopened"):
        await speaker.open()


@pytest.mark.asyncio
async def test_close_without_opening_is_safe_and_idempotent() -> None:
    """Intent: cleanup paths can always close local adapters exactly once."""

    input_module, input_backend = fake_module()
    output_module, output_backend = fake_module()
    microphone = InputMicStream(_pyaudio_module=input_module)
    speaker = OutputSpeakerStream(_pyaudio_module=output_module)

    await microphone.close()
    await microphone.close()
    await speaker.close()
    await speaker.close()

    assert input_backend.terminate_calls == 1
    assert output_backend.terminate_calls == 1


@pytest.mark.asyncio
async def test_speaker_context_writes_exact_pcm_off_the_event_loop() -> None:
    """Intent: playback preserves bytes and delegates blocking PortAudio work to a worker."""

    stream = FakePortAudioStream()
    module, backend = fake_module(stream)
    event_loop_thread = threading.get_ident()
    speaker = OutputSpeakerStream(
        sample_rate=48000,
        channels=2,
        chunk_size=4,
        device_index=1,
        _pyaudio_module=module,
    )
    frame = AudioChunk(
        audio=b"\x01\x00\x02\x00\x03\x00\x04\x00",
        rate=48000,
        width=2,
        channels=2,
    )

    async with speaker:
        await speaker.write(frame)

    assert speaker.sample_rate == 48000
    assert speaker.sample_width == 2
    assert speaker.channels == 2
    assert stream.writes == [frame.audio]
    assert backend.open_calls == [
        {
            "format": 8,
            "channels": 2,
            "rate": 48000,
            "output": True,
            "output_device_index": 1,
            "frames_per_buffer": 4,
        }
    ]
    assert backend.thread_ids["open"][0] != event_loop_thread
    assert stream.thread_ids["write"][0] != event_loop_thread
    assert stream.stop_calls == stream.close_calls == backend.terminate_calls == 1


@pytest.mark.parametrize(
    "frame",
    [
        AudioChunk(audio=b"\x00\x00", rate=8000, width=2, channels=1),
        AudioChunk(audio=b"\x00", rate=16000, width=1, channels=1),
        AudioChunk(audio=b"\x00\x00\x00\x00", rate=16000, width=2, channels=2),
    ],
)
@pytest.mark.asyncio
async def test_speaker_rejects_each_incompatible_pcm_dimension(frame: AudioChunk) -> None:
    """Intent: playback never sends rate, width, or channel mismatches to the device."""

    module, backend = fake_module()
    speaker = OutputSpeakerStream(_pyaudio_module=module)
    await speaker.open()

    with pytest.raises(ValueError, match="mismatch"):
        await speaker.write(frame)

    assert backend.stream.writes == []
    await speaker.close()


@pytest.mark.asyncio
async def test_write_from_propagates_cancellation_after_writing_prior_audio() -> None:
    """Intent: task cancellation reaches the caller instead of being swallowed by playback."""

    module, backend = fake_module()
    speaker = OutputSpeakerStream(_pyaudio_module=module)
    frame = AudioChunk(audio=b"\x01\x00", rate=16000, width=2, channels=1)

    async def cancelled_source() -> AsyncIterator[AudioChunk]:
        yield frame
        raise asyncio.CancelledError

    await speaker.open()
    with pytest.raises(asyncio.CancelledError):
        await speaker.write_from(cancelled_source())

    assert backend.stream.writes == [frame.audio]
    await speaker.close()


@pytest.mark.asyncio
async def test_close_releases_backend_when_stopping_stream_fails() -> None:
    """Intent: a PortAudio stop error cannot leak the stream handle or backend."""

    stream = FakePortAudioStream()
    stream.raise_on_stop = True
    module, backend = fake_module(stream)
    microphone = InputMicStream(_pyaudio_module=module)
    await microphone.open()

    with pytest.raises(OSError, match="stop failed"):
        await microphone.close()

    assert stream.close_calls == 1
    assert backend.terminate_calls == 1
    with pytest.raises(RuntimeError, match="not open"):
        await microphone.read()
