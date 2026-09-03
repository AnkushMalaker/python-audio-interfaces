"""Adapters for applications that intentionally speak the Wyoming protocol."""

from typing import Any

from easy_audio_interfaces.types.audio import AudioChunk


def audio_chunk_to_event(chunk: AudioChunk) -> Any:
    """Convert a native PCM chunk to a Wyoming ``audio-chunk`` event.

    Wyoming is deliberately imported only when this adapter is called. Install
    ``easy-audio-interfaces[wyoming]`` in applications that use this protocol.
    """

    try:
        from wyoming.audio import AudioChunk as WyomingAudioChunk
    except ModuleNotFoundError as error:
        if error.name == "wyoming":
            raise ModuleNotFoundError(
                "Wyoming support is optional; install easy-audio-interfaces[wyoming]"
            ) from error
        raise

    return WyomingAudioChunk(
        audio=chunk.audio,
        rate=chunk.rate,
        width=chunk.width,
        channels=chunk.channels,
        timestamp=chunk.timestamp,
    ).event()


__all__ = ["audio_chunk_to_event"]
