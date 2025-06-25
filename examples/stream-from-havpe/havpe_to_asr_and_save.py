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
from typing import Optional

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioStart
from wyoming.client import AsyncClient

from easy_audio_interfaces import RollingFileSink
from easy_audio_interfaces.network.network_interfaces import TCPServer

DEFAULT_PORT = 8989
SAMP_RATE = 16000
CHANNELS = 1
SAMP_WIDTH = 2  # bytes (16-bit)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ASR_URL = "tcp://192.168.0.110:8765"


class ESP32TCPServer(TCPServer):
    """
    A TCP server for ESP32 devices streaming 32-bit stereo audio.

    Handles the specific format used by ESPHome voice_assistant component:
    - 32-bit little-endian samples (S32_LE)
    - 2 channels (stereo, left/right interleaved)
    - 16kHz sample rate
    - Channel 0 (left) contains processed voice
    - Channel 1 (right) is unused/muted

    The server extracts the left channel and converts from 32-bit to 16-bit
    following the official Home Assistant approach.
    """

    def __init__(self, *args, **kwargs):
        # Set default parameters for ESP32 Voice Kit
        kwargs.setdefault("sample_rate", 16000)
        kwargs.setdefault("channels", 2)
        kwargs.setdefault("sample_width", 4)  # 32-bit = 4 bytes
        super().__init__(*args, **kwargs)

    async def read(self) -> Optional[AudioChunk]:
        """
        Read audio data from the ESP32 TCP client.

        Converts 32-bit stereo data to 16-bit mono by:
        1. Reading raw 32-bit little-endian data
        2. Reshaping to stereo pairs
        3. Extracting left channel (channel 0)
        4. Converting from 32-bit to 16-bit by right-shifting 16 bits

        Returns:
            AudioChunk with 16-bit mono audio, or None if no data/connection closed
        """
        # Get the raw audio chunk from the parent class
        chunk = await super().read()
        if chunk is None:
            return None

        raw_data = chunk.audio

        # Handle empty data
        if len(raw_data) == 0:
            return None

        # Ensure we have complete 32-bit samples (multiple of 8 bytes for stereo)
        if len(raw_data) % 8 != 0:
            logger.warning(
                f"Received incomplete audio frame: {len(raw_data)} bytes, truncating to nearest complete frame"
            )
            raw_data = raw_data[: len(raw_data) - (len(raw_data) % 8)]

        if len(raw_data) == 0:
            return None

        try:
            # Official Home Assistant approach:
            # 1. Parse as 32-bit little-endian integers
            pcm32 = np.frombuffer(raw_data, dtype="<i4")  # 32-bit little-endian

            # 2. Reshape to stereo pairs and extract left channel (channel 0)
            pcm32 = pcm32.reshape(-1, 2)[:, 0]  # Take LEFT channel only

            # 3. Convert from 32-bit to 16-bit by dropping padding and lower bits
            pcm16 = (pcm32 >> 16).astype(np.int16)  # Right shift 16 bits

            # Convert back to bytes
            audio_bytes = pcm16.tobytes()

            return AudioChunk(
                audio=audio_bytes,
                rate=self._sample_rate,
                channels=1,  # Output is mono (left channel only)
                width=2,  # 16-bit = 2 bytes
            )

        except Exception as e:
            logger.error(f"Error processing ESP32 audio data: {e}")
            return None


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
