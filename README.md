# Easy Audio Interfaces

Easy Audio Interfaces is a Python library that provides a simple and flexible way to work with audio streams, including recording, playback, network transfer, and processing.

## Features

- Socket-based audio streaming
- Local file reading and writing
- Audio resampling and rechunking
- Voice activity detection (VAD) using Silero VAD model
- Network file transfer

## Quick Start

Here's a simple example to get you started - record audio from a socket, process it, and save to a file:

```python
from easy_audio_interfaces import SocketReceiver, LocalFileSink, RechunkingBlock, ResamplingBlock

async with SocketReceiver() as receiver, LocalFileSink("output.wav") as sink:
    rechunker = RechunkingBlock(chunk_size=512)
    resampler = ResamplingBlock(original_sample_rate=receiver.sample_rate, resample_rate=16000)

    rechunked_stream = rechunker.rechunk(receiver)
    resampled_stream = resampler.resample(rechunked_stream)
    await sink.write_from(resampled_stream)
```

## Installation

### From PyPI
```bash
uv add easy-audio-interfaces
```

### From Source
```bash
uv add "https://github.com/AnkushMalaker/python-audio-interfaces.git"
```

### Optional Dependencies

Based on the functionality you require, you should consider installing with the following extras:

```bash
# For speech-to-text
uv add "easy-audio-interfaces[stt]"

# For voice activity detection
uv add "easy-audio-interfaces[silero-vad]"

# For Bluetooth audio
uv add "easy-audio-interfaces[bluetooth]"

# For local audio devices
uv add "easy-audio-interfaces[local-audio]"
```

## Usage

### Main Components

#### Audio Sources
- **SocketReceiver**: Receives audio data over a WebSocket connection
- **LocalFileStreamer**: Streams audio data from a local file

#### Audio Sinks
- **SocketStreamer**: Sends audio data over a WebSocket connection
- **LocalFileSink**: Writes audio data to a local file

#### Processing Blocks
- **CollectorBlock**: Collects audio samples for a specified duration
- **ResamplingBlock**: Resamples audio to a different sample rate
- **RechunkingBlock**: Rechunks audio data into fixed-size chunks

#### Voice Activity Detection
- **SileroVad**: Uses the Silero VAD model for voice activity detection
- **VoiceGate**: Applies voice activity detection to segment audio

### Examples

#### Basic Friend Recorder
Records voice segments from a network stream using VAD:

```bash
python -m easy_audio_interfaces.examples.basic_friend_recorder
```

#### File Network Transfer
Transfer audio files over a network:

```bash
# Sender
python -m easy_audio_interfaces.examples.file_network_transfer sender input_file.wav --host localhost --port 8080

# Receiver
python -m easy_audio_interfaces.examples.file_network_transfer receiver output_file.wav --host 0.0.0.0 --port 8080
```

For more detailed usage and API documentation, please refer to the docstrings in the source code.
