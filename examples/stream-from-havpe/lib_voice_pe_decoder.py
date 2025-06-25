#!/usr/bin/env python3
"""
Simple TCP Server for ESP32 32-bit stereo audio.

This module provides ESP32TCPServer, a subclass of TCPServer that handles
32-bit stereo audio from ESP32 following the official Home Assistant approach.

The ESP32 sends 32-bit little-endian stereo audio where:
- Channel 0 (left): Processed voice from XMOS pipeline (AEC/noise suppression)
- Channel 1 (right): Unused/muted (set to PIPELINE_STAGE_NONE)

This implementation extracts the left channel and converts from 32-bit to 16-bit.
"""

import logging
from typing import Optional

import numpy as np
from wyoming.audio import AudioChunk

from easy_audio_interfaces.network.network_interfaces import TCPServer

logger = logging.getLogger(__name__)


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
