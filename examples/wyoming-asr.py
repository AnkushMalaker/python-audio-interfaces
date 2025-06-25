import asyncio
import logging
import time

from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart, AudioStop
from wyoming.client import AsyncClient

from easy_audio_interfaces.extras.local_audio import InputMicStream

logging.basicConfig(level=logging.INFO)


async def main():
    stream = InputMicStream(
        sample_rate=16000,
        channels=1,
    )
    client = AsyncClient.from_uri("tcp://192.168.0.110:8765")
    await client.connect()
    await client.write_event(Transcribe().event())
    await client.write_event(AudioStart(rate=16000, width=2, channels=1).event())

    st = time.time()

    await stream.open()
    logging.info("Mic is working")
    async for chunk in stream:
        elapsed = time.time() - st
        if elapsed > 10:
            break
        print(f"Sending chunk {elapsed:.2f}s")
        await client.write_event(chunk.event())

    await client.write_event(AudioStop().event())
    while True:
        print("Waiting for transcript")
        k = await client.read_event()
        print(k)
        if k and Transcript.is_type(k.type):
            logging.info(k)
            break

    await client.disconnect()
    await stream.close()


if __name__ == "__main__":
    asyncio.run(main())
