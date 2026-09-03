# Easy Audio Interfaces

Easy Audio Interfaces is a Python library that provides a simple and flexible way to work with audio streams, including recording, playback, network transfer, and processing.

## Features

- Socket-based audio streaming
- Local file reading and writing
- Audio resampling and rechunking
- Network file transfer
- Native, validated PCM audio chunks with no protocol dependency

## Quick Start

Here's a simple example that streams a WAV file, resamples it, and writes the result:

```python
from easy_audio_interfaces import LocalFileSink, LocalFileStreamer, ResamplingBlock

async with LocalFileStreamer("input.wav", chunk_size_ms=20) as source:
    resampler = ResamplingBlock(resample_rate=16_000)
    async with resampler, LocalFileSink(
        "output.wav", sample_rate=16_000, channels=source.channels
    ) as sink:
        await sink.write_from(resampler.process(source))
```

### Audio Model

`AudioChunk` and `AudioFormat` are native, protocol-neutral PCM types. A chunk contains raw
interleaved audio bytes plus its sample rate, bytes per sample, channel count, and optional
timestamp. Construction validates the format and rejects partial sample frames. Protocol and
transport adapters are intentionally outside this library.

#### Advanced Usage: Manual Chunk Processing with ResamplingBlock

For more control over individual audio chunks, `process_chunk` keeps one filter state across
calls. Finalize it once, at the logical end of the stream, with `process_chunk_last`:

```python
from easy_audio_interfaces import ResamplingBlock
resampler = ResamplingBlock(resample_rate=16000)
await resampler.open()

# Process individual chunks
for chunk in audio_chunks:
    async for resampled_chunk in resampler.process_chunk(chunk):
        # Handle each resampled chunk
        process_audio(resampled_chunk)

# Important: Flush remaining buffered samples
async for final_chunk in resampler.process_chunk_last():
    process_audio(final_chunk)

await resampler.close()
```

### Stateful sample-rate conversion

Code that does not need the async processing-block adapter can use `SampleRateConverter`
directly. Its scope is intentionally narrow: it changes the rate of signed 16-bit or 32-bit
little-endian PCM while preserving width and channel topology.

```python
from easy_audio_interfaces import SampleRateConverter

converter = SampleRateConverter(24_000)
for chunk in source_chunks:
    converted = converter.push(chunk)
    if converted is not None:
        consume(converted)

remaining = converter.finish()
if remaining is not None:
    consume(remaining)
```

`push()` may return `None` while the filter is buffering. `finish()` drains that latency and is
idempotent. Input format changes and pushes after finalization fail explicitly; call `reset()`
before beginning another logical stream. The converter does not own timestamps, sequence
numbers, packet sizing, codecs, or transport acknowledgements—those remain application concerns.

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
- **SocketServer**: Receives audio messages over a WebSocket connection
- **TCPServer**: Receives raw audio bytes over TCP
- **LocalFileStreamer**: Streams audio data from a local file
- **SourceFromBytesIO**: Streams a WAV held in memory

#### Audio Sinks
- **SocketClient**: Sends audio messages over a WebSocket connection
- **TCPClient**: Sends raw audio bytes over TCP
- **LocalFileSink**: Writes audio data to a local file
- **RollingFileSink**: Splits output across time-based WAV segments

#### Processing Blocks
- **SampleRateConverter**: Stateful synchronous PCM rate conversion with explicit finalization
- **ResamplingBlock**: Resamples audio to a different sample rate
  - `process_chunk_last()`: Flushes remaining buffered samples from the resampler.
    Call this after processing all chunks to ensure no audio data is lost due to internal buffering.
- **RechunkingBlock**: Rechunks audio data into fixed-size chunks

### Examples

#### File Network Transfer
Transfer audio files over a network:

```bash
# Sender
python examples/file_network_transfer.py sender input_file.wav --host localhost --port 8080

# Receiver
python examples/file_network_transfer.py receiver output_file.wav --host 0.0.0.0 --port 8080
```

For more detailed usage and API documentation, please refer to the docstrings in the source code.
