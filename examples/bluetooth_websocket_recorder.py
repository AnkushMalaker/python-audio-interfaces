import argparse
import asyncio
import logging
from pathlib import Path

from opuslib import Decoder

from easy_audio_interfaces.audio_interfaces import (
    LocalFileSink,
    SocketReceiver,
    SocketStreamer,
)
from easy_audio_interfaces.extras.friend_bluetooth import (
    AUDIO_DATA_STREAM_UUID,
    DEVICE_NAME,
    SERVICE_UUID,
    FriendFrameProcessor,
    find_device_by_name,
)
from easy_audio_interfaces.types import NumpyFrame

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RECORDINGS_DIR = Path("./recordings")
RECORDINGS_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SIZE = 960

decoder = Decoder(SAMPLE_RATE, CHANNELS)


async def bluetooth_client(host, port):
    frame_processor = FriendFrameProcessor(SAMPLE_RATE, CHANNELS)
    device = await find_device_by_name(DEVICE_NAME)
    logger.info(f"Found device: {device}")
    if device is None:
        logger.error(f"Device with the name '{DEVICE_NAME}' not found.")
        return

    async with SocketStreamer(host=host, port=port) as streamer:
        from bleak import BleakClient

        async with BleakClient(device) as client:
            logger.info(f"Connected to Bluetooth device: {client.is_connected}")

            def audio_data_handler(characteristic, data):
                frame_processor.store_frame_packet(data)

            await client.start_notify(AUDIO_DATA_STREAM_UUID, audio_data_handler)

            try:
                while client.is_connected:
                    pcm_data = await frame_processor.decode_frames()
                    if pcm_data:  # Only process non-empty frames
                        numpy_frame = NumpyFrame.frombuffer(pcm_data)
                        await streamer.write(numpy_frame)
                    else:
                        await asyncio.sleep(0.1)  # Add a small delay to prevent tight loop
            finally:
                await client.stop_notify(AUDIO_DATA_STREAM_UUID)
                logger.info("Disconnected from Bluetooth device")


async def websocket_server(host, port):
    receiver = SocketReceiver(
        host=host,
        port=port,
        sample_rate=SAMPLE_RATE,
        post_process_bytes_fn=lambda x: NumpyFrame.frombuffer(x),
    )
    await receiver.open()

    async with (
        LocalFileSink(
            RECORDINGS_DIR / "output.wav", sample_rate=SAMPLE_RATE, channels=CHANNELS
        ) as sink
    ):
        await sink.write_from(receiver)


async def main():
    parser = argparse.ArgumentParser(description="Bluetooth WebSocket Recorder")
    parser.add_argument(
        "--mode",
        default="c",
        choices=["c", "s"],
        help="Run as client (c) or server (s)",
        required=False,
    )
    parser.add_argument("--host", default="localhost", help="WebSocket host (default: localhost)")
    parser.add_argument("--port", type=int, default=8081, help="WebSocket port (default: 8081)")
    args = parser.parse_args()

    if args.mode == "c":
        await bluetooth_client(args.host, args.port)
    else:
        await websocket_server(args.host, args.port)


if __name__ == "__main__":
    asyncio.run(main())
