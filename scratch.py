import argparse
import asyncio
import logging

from easy_audio_interfaces.audio_interfaces import (
    LocalFileSink,
    SocketReceiver,
    SocketStreamer,
)
from easy_audio_interfaces.examples.basic_friend_recorder import decode_friend_message
from easy_audio_interfaces.extras.local_audio import InputMicStream, OutputSpeakerStream

logger = logging.getLogger(__name__)


async def main():
    receiver = SocketReceiver(
        host="0.0.0.0",
        port=8081,
        sample_rate=16000,
        channels=1,
        post_process_bytes_fn=decode_friend_message,
    )
    # sink = LocalFileSink(file_path="output.wav", sample_rate=16000, channels=1)

    await receiver.open()
    # await sink.open()

    async with OutputSpeakerStream() as speaker:
        try:
            async for frame in receiver:
                print("writing frame ")
                await speaker.write(frame)
        finally:
            await receiver.close()


if __name__ == "__main__":
    asyncio.run(main())
