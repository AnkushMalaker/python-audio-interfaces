import asyncio
import logging

from easy_audio_interfaces.extras.local_audio import InputMicStream
from easy_audio_interfaces.network import SocketClient

logging.basicConfig(level=logging.INFO)


async def client():
    client = SocketClient(host="localhost", port=4369)
    async with InputMicStream() as mic_source:
        async with client as client:
            async for chunk in mic_source:
                await client.write(chunk)


if __name__ == "__main__":
    asyncio.run(client())
