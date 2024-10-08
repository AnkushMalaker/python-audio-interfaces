import asyncio
import logging
from pathlib import Path

import numpy as np
from opuslib import Decoder
from scipy.io import wavfile

from easy_audio_interfaces.audio_interfaces import RechunkingBlock, SocketReceiver
from easy_audio_interfaces.extras.models import SileroVad, VoiceGate
from easy_audio_interfaces.types import NumpyFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path("./recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

decoder = Decoder(fs=16000, channels=1)


def post_process_audio_chunk(chunk: bytes) -> NumpyFrame:
    opus_data = chunk[3:]
    pcm_data = decoder.decode(bytes(opus_data), frame_size=960)
    return NumpyFrame(np.frombuffer(pcm_data, dtype=np.int16))


async def main():
    host = "0.0.0.0"
    port = 8080

    receiver = SocketReceiver(
        host=host, port=port, sample_rate=16000, post_process_callback=post_process_audio_chunk
    )
    await receiver.open()
    rechunking_block = RechunkingBlock(chunk_size=512)
    silero_vad = SileroVad(sampling_rate=receiver.sample_rate)
    voice_gate = VoiceGate(starting_patience=5, stopping_patience=20, cool_down=20, threshold=0.1)

    segment_counter = 0
    try:
        chunk_iterator = rechunking_block.rechunk(receiver)
        async for voice_segment in silero_vad.iter_segments(chunk_iterator, voice_gate):
            segment_counter += 1
            filename = RECORDINGS_DIR / f"voice_segment_{segment_counter}.wav"
            wavfile.write(filename, 16000, voice_segment)
            logger.info(f"Saved voice segment to {filename}")
    finally:
        await receiver.close()


if __name__ == "__main__":
    asyncio.run(main())
