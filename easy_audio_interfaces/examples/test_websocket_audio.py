import asyncio
import logging
from pathlib import Path

import numpy as np
import websockets
from opuslib import Decoder
from scipy.io import wavfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path("./recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

decoder = Decoder(fs=16000, channels=1)


def post_process_audio_chunk(chunk: bytes) -> np.ndarray:
    opus_data = chunk[3:]
    pcm_data = decoder.decode(bytes(opus_data), frame_size=160)
    return np.frombuffer(pcm_data, dtype=np.int16)


async def receive_audio(websocket):
    logger.info("Client connected")
    accumulated_samples = []

    try:
        async for message in websocket:
            audio_data = post_process_audio_chunk(message)
            accumulated_samples.append(audio_data)
            logger.debug(f"Received chunk with {len(audio_data)} samples")
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        if accumulated_samples:
            audio_data = np.concatenate(accumulated_samples)
            filename = RECORDINGS_DIR / "test-websocket.wav"
            wavfile.write(filename, 16000, audio_data)
            logger.info(f"Saved audio to {filename}")


async def main():
    server = await websockets.serve(receive_audio, "0.0.0.0", 8080)
    logger.info("WebSocket server started on ws://0.0.0.0:8080")
    await server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
