# Easy audio interfaces for python
# Introduction
I found that there are many important but hard to get started with concepts that go into voice systems. For example, the understanding chunk sizes in recording audio, resampling, blah blah. But for a person who wants to build an app that takes user words as input, they don't need to know any of that. They should be able to say, take the stream of input audio, find voice, if voice, tell me the text, give me  that text. Don't complicate it. Lets say your application just wants to hear for "send screenshot" and then take a screenshot and send it to the your coding partner. Perhaps here you'd want to listen to audio stream using `InputMicStream()`, combine it with a `WhisperBlockForASR()`. Using `chaining`, you can chain the output of `InputMicStream()` to `WhisperBlockForASR()` and pass the output of that to a condition. This package makes it simple to do that by
```python
from audio_interfaces import InputMicStream, WhisperBlockForASR

# Focus on your application logic
def take_screenshot():
    ...

def send_screenshot():
    ...

# Context managers handle closing and opening streams, handling cleanup if saving files required etc.
with (
        InputMicStream() as micstream,
        WhisperBlockForASR() as whisper
    ):
        whisper.processor_from(mic_stream) # Configure whisper to use mic_stream. Any conversion of sample_rate etc is handled automatically by the above config.
        whisper.write_from(mic_stream)
        # WhisperBlockForASR implements __iter__ so you can iterate over it
        # Queues handle the rest so you can simply focus on logic
        for user_said in whisper:
            if user_said == "send screenshot":
                take_screenshot()
                send_screenshot()
# Check examples/asr_over_network.py for using a raspberry pi to stream data from one room and process in another room in few lines of code.
```

# Installation
## Pre-requisites
Install portaudio if not already installed
```sudo apt install portaudio19-dev```
## Install this package
```pip install git+https://github.com/AnkushMalaker/python-audio-interfaces.git```

# Usage
Look at examples for more details

The basic usage of this package  is to import existing implementations of audio interfaces from this package or import protocols to subclass from and make your  functionality compatible with easy_audio_interfaces so that you can use the rest of the eco-system.

# Existing Interfaces
## General Interfaces
- [x] AudioStream
- [x] AudioSink
- [x] AudioProcessingBlock
## Specific Interfaces
- [x] InputMicStream
- [x] OutputSpeakerStream
- [x] OutputFileStream
- [x] InputFileStream
- [x] SocketStream
- [x] CollectorInterface
- [x] WhisperASRBlock
