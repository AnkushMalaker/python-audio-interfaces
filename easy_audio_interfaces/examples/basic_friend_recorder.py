import asyncio
import logging
from pathlib import Path

import fire
import numpy as np
from scipy.io import wavfile

from easy_audio_interfaces.audio_interfaces import CollectorBlock, SocketReceiver
from easy_audio_interfaces.types.audio import NumpyFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path("./recordings")

from opuslib import Decoder

decoder = Decoder(fs=16000, channels=1)


def post_process_friend_audio_chunk(chunk: bytes) -> NumpyFrame:
    opus_data = chunk[3:]
    pcm_data = decoder.decode(bytes(opus_data), frame_size=160)
    return NumpyFrame.frombuffer(pcm_data)


async def record_websocket_audio(host: str = "0.0.0.0", port: int = 8080, chunk_duration: int = 10):
    RECORDINGS_DIR.mkdir(exist_ok=True)

    receiver = SocketReceiver(
        host=host,
        port=port,
        sample_rate=16000,
        channels=1,
        post_process_callback=post_process_friend_audio_chunk,
    )

    await receiver.open()

    try:
        logger.info(f"Listening for WebSocket connections on ws://{host}:{port}")
        collector = CollectorBlock(sample_rate=receiver.sample_rate, collect_seconds=chunk_duration)

        chunk_count = 0
        while True:  # Run indefinitely
            async for waveform in collector.collect(receiver):
                chunk_count += 1
                filename = RECORDINGS_DIR / f"chunk_{chunk_count:04d}.wav"

                # Convert to int16 for WAV file
                audio_data = (waveform.normalize() * 32767).astype(np.int16)

                wavfile.write(filename, receiver.sample_rate, audio_data)
                logger.info(f"Saved {filename}")
    finally:
        await receiver.close()


def main(host: str = "0.0.0.0", port: int = 8080, chunk_duration: int = 10):
    asyncio.run(record_websocket_audio(host, port, chunk_duration))


if __name__ == "__main__":
    fire.Fire(main)
