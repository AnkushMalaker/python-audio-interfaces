import asyncio
import logging
from typing import Optional

import fire

from easy_audio_interfaces.audio_interfaces import CollectorBlock, SocketReceiver
from easy_audio_interfaces.extras.models import WhisperBlock

logging.basicConfig(level=logging.ERROR)


async def main_async(models_root: Optional[str] = None):
    model = WhisperBlock(model_description="distil-large-v3", models_root=models_root)
    async with SocketReceiver(sample_rate=16000) as receiver:
        collector = CollectorBlock(sample_rate=receiver.sample_rate, collect_seconds=5)
        async for waveform in collector.collect(receiver):
            waveformf32 = waveform.normalize()
            segments, _ = model.transcribe(
                waveformf32,
                vad_filter=True,
            )
            segments = list(segments)
            if segments:
                print("Transcript:")
                for segment in segments:
                    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")


def main(models_root: Optional[str] = None):
    asyncio.run(main_async(models_root))


if __name__ == "__main__":
    fire.Fire(main)
