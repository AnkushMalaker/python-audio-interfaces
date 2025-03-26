import asyncio
import logging
import time

from easy_audio_interfaces.extras.local_audio import InputMicStream
from easy_audio_interfaces.filesystem import LocalFileSink

logging.basicConfig(level=logging.INFO)


async def record_local_audio(duration: int = 5, output_file: str = "recorded_audio.wav"):
    """
    Record audio from the local microphone for a specified duration and save it to a file.

    Args:
        duration (int): The duration of the recording in seconds. Default is 5 seconds.
        output_file (str): The name of the output file. Default is "recorded_audio.wav".
    """
    mic_source = InputMicStream()

    file_sink = LocalFileSink(
        output_file, sample_rate=mic_source.sample_rate, channels=mic_source.channels
    )

    print(f"Recording audio for {duration} seconds...")
    start_time = time.time()

    await mic_source.open()

    async with file_sink as sink:
        async for chunk in mic_source:
            await sink.write(chunk)
            if time.time() - start_time > duration:
                break

    await mic_source.close()

    print(f"Audio recorded and saved to {output_file}")

    print(f"Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    asyncio.run(record_local_audio())
