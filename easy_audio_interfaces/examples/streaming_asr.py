import asyncio
import logging
import os
from typing import Optional

import fire

from easy_audio_interfaces.audio_interfaces import (
    CollectorBlock,
    ResamplingBlock,
    SocketReceiver,
)
from easy_audio_interfaces.extras.models import WhisperBlock
from easy_audio_interfaces.types import NumpyFrame

logging.basicConfig(level=logging.ERROR)


async def main_async(models_root: Optional[str] = os.environ.get("MODELS_DIR")):
    model = WhisperBlock(model_description="distil-large-v3", models_root=models_root)
    async with SocketReceiver(sample_rate=16000) as receiver:
        resampler = ResamplingBlock(original_sample_rate=receiver.sample_rate, resample_rate=16000)
        collector = CollectorBlock(sample_rate=16000, collect_seconds=5)
        resampled_stream = resampler.resample(receiver)
        async for waveform in collector.collect(resampled_stream):
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


def main(models_root: str = os.environ.get("MODELS_DIR", "./Models")):
    asyncio.run(main_async(models_root))


if __name__ == "__main__":
    fire.Fire(main)
