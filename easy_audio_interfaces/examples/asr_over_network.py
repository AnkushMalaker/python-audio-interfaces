import asyncio
import logging
import os

import fire

from easy_audio_interfaces.audio_interfaces import (
    CollectorBlock,
    SocketReceiver,
    SocketStreamer,
)
from easy_audio_interfaces.extras.models import WhisperBlock

logging.basicConfig(level=logging.ERROR)


async def main_async(role: str, models_root: str = os.environ.get("MODELS_DIR", "./Models")):
    transmission_sr = 16000
    if role == "server":
        model = WhisperBlock(model_description="medium.en", models_root=models_root)
        async with SocketReceiver(
            host="0.0.0.0", port=5025, sample_rate=transmission_sr, channels=1
        ) as receiver:
            collector = CollectorBlock(sample_rate=transmission_sr, collect_seconds=5)
            async for waveform in collector.collect(receiver):
                waveformf32 = waveform.normalize()
                segments, _ = model.transcribe(
                    waveformf32,
                    vad_filter=True,
                )
                segments = list(segments)
                if len(segments) > 0:
                    print("Transcript:")
                    for segment in segments:
                        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
    elif role == "client":
        streamer = SocketStreamer(port=5025, sample_rate=transmission_sr, channels=1)
        # Replace the following line with your actual audio source
        audio_source = get_audio_source()
        await streamer.write_from(audio_source)
        await streamer.close()  # Assuming there's a close method
    else:
        raise ValueError("role must be 'server' or 'client'")


def main(role: str, models_root: str = os.environ.get("MODELS_DIR", "./Models")):
    asyncio.run(main_async(role, models_root))


if __name__ == "__main__":
    fire.Fire(main)
