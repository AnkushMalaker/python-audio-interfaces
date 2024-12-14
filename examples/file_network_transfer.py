import asyncio
import logging

import fire

from easy_audio_interfaces import LocalFileSink, LocalFileStreamer
from easy_audio_interfaces.network.network_interfaces import (
    SocketReceiver,
    SocketStreamer,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def sender(input_file: str, host: str = "localhost", port: int = 8080):
    logger.info(f"Sending file: {input_file}")
    async with LocalFileStreamer(input_file) as file_source:
        async with SocketStreamer(
            sample_rate=file_source.sample_rate,
            channels=file_source.channels,
            host=host,
            port=port,
        ) as streamer:
            await streamer.write_from(file_source)
    logger.info(f"Finished sending file to {host}:{port}")


async def receiver(
    output_file: str, host: str = "0.0.0.0", port: int = 8080, timeout: float = 30.0
):
    logger.info(f"Receiving audio and saving to: {output_file}")
    async with SocketReceiver(host=host, port=port) as socket_source:
        logger.info(
            f"SocketReceiver opened. Sample rate: {socket_source.sample_rate}, Channels: {socket_source.channels}"
        )
        async with LocalFileSink(
            output_file,
            sample_rate=socket_source.sample_rate,
            channels=socket_source.channels,
        ) as file_sink:
            logger.info(f"LocalFileSink opened. Writing to {output_file}")
            try:
                await asyncio.wait_for(file_sink.write_from(socket_source), timeout=timeout)
            except asyncio.TimeoutError:
                logger.info(f"Receiver timed out after {timeout} seconds")
            finally:
                await socket_source.stop()
    logger.info(f"Finished receiving and saving file to {output_file}")


async def main_async(role: str, file_path: str, host: str = "localhost", port: int = 8080):
    if role == "sender":
        await sender(file_path, host, port)
    elif role == "receiver":
        await receiver(file_path, host, port)
    else:
        raise ValueError("Role must be 'sender' or 'receiver'")


def main(
    role: str, file_path: str, host: str = "localhost", port: int = 8080, timeout: float = 10.0
):
    if role == "sender":
        asyncio.run(sender(file_path, host, port))
    elif role == "receiver":
        asyncio.run(receiver(file_path, host, port, timeout))
    else:
        raise ValueError("Role must be 'sender' or 'receiver'")


if __name__ == "__main__":
    fire.Fire(main)
