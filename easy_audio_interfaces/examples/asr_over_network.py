import logging
import os

import fire

from easy_audio_interfaces.audio_interfaces import (
    CollectorBlock,
    InputMicStream,
    OutputFileStream,
    ResamplingBlock,
    SocketReceiver,
    SocketStreamer,
)
from easy_audio_interfaces.extras.models import WhisperBlock

logging.basicConfig(level=logging.ERROR)


def save_to_file(waveform, file_name, sr):
    output_file_stream = OutputFileStream(file_path=file_name, sample_rate=sr)
    output_file_stream.open()
    output_file_stream.write(waveform)
    output_file_stream.close()


def main(role: str, models_root: str = os.environ.get("MODELS_DIR", "./Models")):
    transmission_sr = 16000
    if role == "server":
        model = WhisperBlock(model_description="medium.en", models_root=models_root)
        with (
            SocketReceiver(
                host="0.0.0.0", port=5025, sample_rate=transmission_sr, channels=1
            ) as reciever,
            CollectorBlock(sample_rate=transmission_sr, collect_seconds=5) as collector,
        ):
            collector.write_from(reciever)
            for waveform in collector:
                waveformf32 = waveform.normalize()
                # Need to normalize it if model type is float32 or to use vad_filter

                segments, _ = model.transcribe(
                    waveformf32,
                    vad_filter=True,
                    # condition_on_previous_text=False,
                )
                segments = list(segments)
                if len(segments) > 0:
                    print("Transcript:")
                    for segment in segments:
                        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    elif role == "client":
        with (
            InputMicStream() as audio_stream,
            ResamplingBlock(
                original_sample_rate=audio_stream.sample_rate, resample_rate=transmission_sr
            ) as resampler,
            SocketStreamer(port=5025, sample_rate=transmission_sr, channels=1) as streamer,
        ):
            resampler.write_from(audio_stream)
            for segment in resampler:
                streamer.write(segment)
    else:
        raise ValueError("role must be 'server' or 'client'")


if __name__ == "__main__":
    fire.Fire(main)
