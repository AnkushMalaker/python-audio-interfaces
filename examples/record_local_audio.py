import asyncio
import logging
import time
from pathlib import Path

from easy_audio_interfaces.extras.local_audio import InputMicStream
from easy_audio_interfaces.filesystem import RollingFileSink  # or LocalFileSink

logging.basicConfig(level=logging.INFO)


async def record_local_audio(duration: int = 30, output_dir: str | Path = "recorded_audio"):
    """
    Record audio from the local microphone for a specified duration and save it to a file.

    Args:
        duration (int): The duration of the recording in seconds. Default is 5 seconds.
        output_file (str): The name of the output file. Default is "recorded_audio.wav".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # p = PyAudioInterface()
    # print(p.list_input_devices())
    mic_source = InputMicStream()

    file_sink = RollingFileSink(
        directory=output_dir,
        prefix="",
        segment_duration_seconds=60,
        sample_rate=mic_source.sample_rate,
        channels=mic_source.channels,
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

    print(f"Audio recorded and saved to {output_dir}")

    print(f"Time taken: {time.time() - start_time} seconds")


if __name__ == "__main__":
    asyncio.run(record_local_audio())
