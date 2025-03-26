import asyncio
import logging

from easy_audio_interfaces.extras.local_audio import OutputSpeakerStream
from easy_audio_interfaces.network import SocketServer

logging.basicConfig(level=logging.INFO)


async def serve():
    server = SocketServer(host="0.0.0.0", port=4369)
    output_speaker = OutputSpeakerStream()
    async with server as server, output_speaker as output_speaker:
        async for chunk in server:
            output_speaker.write(chunk)


if __name__ == "__main__":
    asyncio.run(serve())
