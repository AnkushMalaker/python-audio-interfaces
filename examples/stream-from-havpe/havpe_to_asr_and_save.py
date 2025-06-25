#!/usr/bin/env python3
"""
TCP WAV recorder using easy_audio_interfaces library with ESP32 I²S swap detection.

• Listens on PORT (default 8989) for ESP32 client
• Saves audio to rolling file sink
• Forwards audio to ASR server for transcription
• Automatically detects and fixes ESP32 I²S mono sample swap issue
"""

import argparse
import asyncio
import logging
import pathlib

from lib_voice_pe_decoder import ESP32TCPServer
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioStart
from wyoming.client import AsyncClient

from easy_audio_interfaces import RollingFileSink

DEFAULT_PORT = 8989
SAMP_RATE = 16000
CHANNELS = 1
SAMP_WIDTH = 2  # bytes (16-bit)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASR_URL = "tcp://192.168.0.110:8765"


async def transcription_printer(client: AsyncClient):
    """Print transcription results from ASR client."""
    while True:
        k = await client.read_event()
        if k and Transcript.is_type(k.type):
            print(f"Transcription event!: {k}")
        else:
            print("whatthehelly?")
            print(f"Unknown event: {k}")


async def process_esp32_audio(
    esp32_server: ESP32TCPServer, asr_client: AsyncClient, file_sink: RollingFileSink
):
    """Process audio chunks from ESP32 server, save to file sink and send to ASR client."""
    try:
        logger.info("Starting to process ESP32 audio for ASR and file saving...")
        chunk_count = 0
        async for chunk in esp32_server:
            chunk_count += 1
            if chunk_count % 10 == 1:  # Log every 10th chunk
                logger.info(
                    f"Received chunk {chunk_count} from ESP32, size: {len(chunk.audio)} bytes"
                )

            # Write to rolling file sink
            await file_sink.write(chunk)

            # Send to ASR
            await asr_client.write_event(chunk.event())
    except asyncio.CancelledError:
        logger.info("ESP32 audio processor cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in ESP32 audio processor: {e}")
        raise


async def main():
    parser = argparse.ArgumentParser(description="TCP WAV recorder with ESP32 I²S swap detection")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="TCP port to listen on for ESP32 (default 8989)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind to (default 0.0.0.0)",
    )
    parser.add_argument(
        "--segment-duration",
        type=int,
        default=5,
        help="Duration of each audio segment in seconds (default 5)",
    )
    parser.add_argument("-v", "--verbose", action="count", default=0, help="-v: INFO, -vv: DEBUG")
    args = parser.parse_args()

    loglevel = logging.WARNING - (10 * min(args.verbose, 2))
    logging.basicConfig(format="%(asctime)s  %(levelname)s  %(message)s", level=loglevel)

    # Create recordings directory
    recordings = pathlib.Path("recordings")
    recordings.mkdir(exist_ok=True)

    esp32_recordings = pathlib.Path("recordings/esp32_raw")
    esp32_recordings.mkdir(exist_ok=True, parents=True)

    logger.info("Using REAL ASR server")
    # Create ASR client
    asr_client = AsyncClient.from_uri(ASR_URL)
    await asr_client.connect()
    await asr_client.write_event(Transcribe().event())
    await asr_client.write_event(
        AudioStart(rate=SAMP_RATE, width=SAMP_WIDTH, channels=CHANNELS).event()
    )

    # Create ESP32 TCP server with automatic I²S swap detection
    esp32_server = ESP32TCPServer(
        host=args.host,
        port=args.port,
        sample_rate=SAMP_RATE,
        channels=CHANNELS,
        sample_width=4,
    )

    # Create rolling file sink for ESP32 data
    esp32_file_sink = RollingFileSink(
        directory=esp32_recordings,
        prefix="esp32_raw",
        segment_duration_seconds=args.segment_duration,
        sample_rate=SAMP_RATE,
        channels=CHANNELS,
        sample_width=SAMP_WIDTH,
    )

    # Start transcription printer task
    asyncio.create_task(transcription_printer(asr_client))

    try:
        # Start ESP32 server
        async with esp32_server:
            logger.info(f"ESP32 server listening on {args.host}:{args.port}")

            # Create rolling file sink for ESP32 raw data
            async with esp32_file_sink as file_sink:
                logger.info("Starting audio recording and ASR processing...")

                # Start audio processing task
                audio_processor_task = asyncio.create_task(
                    process_esp32_audio(esp32_server, asr_client, file_sink)
                )

                try:
                    # Wait for task to complete or be interrupted
                    await audio_processor_task

                except KeyboardInterrupt:
                    logger.info("Interrupted – stopping recording")
                finally:
                    audio_processor_task.cancel()

                    # Wait for task to complete cancellation
                    await asyncio.gather(audio_processor_task, return_exceptions=True)

                    # Disconnect ASR client
                    await asr_client.disconnect()

    except KeyboardInterrupt:
        logger.info("Interrupted – closing servers")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        logger.info("Recording session ended")


if __name__ == "__main__":
    asyncio.run(main())
