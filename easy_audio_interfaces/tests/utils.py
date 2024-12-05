import asyncio
from typing import AsyncGenerator, cast

from pydub import AudioSegment


async def async_generator(wav_segment: AudioSegment) -> AsyncGenerator[AudioSegment, None]:
    # Split the wav_segment into 1-second chunks
    chunk_duration_ms = 1000
    for i in range(0, len(wav_segment), chunk_duration_ms):
        chunk = cast(AudioSegment, wav_segment[i : i + chunk_duration_ms])
        await asyncio.sleep(0.0)  # simulate async
        yield chunk
