Easy Audio Interfaces
Easy Audio Interfaces is a Python library that provides a simple and flexible way to work with audio streams, including recording, playback, network transfer, and processing.
Features:
Socket-based audio streaming
Local file reading and writing
Audio resampling and rechunking
Voice activity detection (VAD) using Silero VAD model
Network file transfer
Installation:
To install Easy Audio Interfaces, use pip:
pip install easy-audio-interfaces
Usage:
The library provides various audio interfaces and processing blocks that can be used to build audio processing pipelines. Here are some examples:
1. Basic Friend Recorder (examples/basic_friend_recorder.py):
This example demonstrates how to use SocketReceiver, RechunkingBlock, SileroVad, and VoiceGate to record voice segments from a network stream.
To run:
python -m easy_audio_interfaces.examples.basic_friend_recorder
File Network Transfer (examples/file_network_transfer.py):
This example shows how to transfer audio files over a network using SocketStreamer and SocketReceiver.
To run as a sender:
python -m easy_audio_interfaces.examples.file_network_transfer sender input_file.wav --host localhost --port 8080
To run as a receiver:
python -m easy_audio_interfaces.examples.file_network_transfer receiver output_file.wav --host 0.0.0.0 --port 8080
Main Components:
1. Audio Sources:
SocketReceiver: Receives audio data over a WebSocket connection.
LocalFileStreamer: Streams audio data from a local file.
Audio Sinks:
SocketStreamer: Sends audio data over a WebSocket connection.
LocalFileSink: Writes audio data to a local file.
3. Processing Blocks:
CollectorBlock: Collects audio samples for a specified duration.
ResamplingBlock: Resamples audio to a different sample rate.
RechunkingBlock: Rechunks audio data into fixed-size chunks.
Voice Activity Detection:
SileroVad: Uses the Silero VAD model for voice activity detection.
VoiceGate: Applies voice activity detection to segment audio.
To use these components, you can create audio processing pipelines by chaining them together. For example:
async with SocketReceiver() as receiver, LocalFileSink("output.wav") as sink:
rechunker = RechunkingBlock(chunk_size=512)
resampler = ResamplingBlock(original_sample_rate=receiver.sample_rate, resample_rate=16000)
rechunked_stream = rechunker.rechunk(receiver)
resampled_stream = resampler.resample(rechunked_stream)
await sink.write_from(resampled_stream)
This pipeline receives audio from a socket, rechunks it, resamples it to 16kHz, and saves it to a local file.
For more detailed usage and API documentation, please refer to the docstrings in the source code or the generated API documentation.

# Essential Extras
Based on the functinoality you require, you should consider installing with following extras:
optional-dependencies = { stt = [
  "faster-whisper",
], silero-vad = [
  "torch",
  "torchaudio",
], bluetooth = [
  "bleak",
], local-audio = [
  "pyaudio",
] }

# Installing in your project
You can install using uv:
uv add "https://github.com/AnkushMalaker/python-audio-interfaces.git"

or with extras, 

uv add "https://github.com/AnkushMalaker/python-audio-interfaces.git[local-audio]"
