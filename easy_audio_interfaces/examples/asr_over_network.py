import logging
import os
from pathlib import Path
from typing import Any, Optional

import fire
from numpy.typing import NDArray

from easy_audio_interfaces.audio_interfaces import (
    CollectorBlock,
    InputMicStream,
    OutputFileStream,
    ResamplingBlock,
    SocketReceiver,
    SocketStreamer,
)
from easy_audio_interfaces.types import PathLike

logging.basicConfig(level=logging.ERROR)


def save_to_file(waveform, file_name, sr):
    output_file_stream = OutputFileStream(file_path=file_name, sample_rate=sr)
    output_file_stream.open()
    output_file_stream.write(waveform)
    output_file_stream.close()


class WhisperBlock:
    def __init__(
        self,
        model_description: str = "guillaumekln/faster-whisper-large-v2",
        language: Optional[str] = None,
        models_root: PathLike = Path("/home/models"),
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("Please install stt feature to use the WhisperBlock")
        self.language = language or "en"
        self.model = WhisperModel(
            model_description,
            download_root=str(models_root),
        )
        self.feature_extractor = self.model.feature_extractor

    def open(self):
        pass

    def close(self):
        pass

    def transcribe(self, audio: NDArray, **kwargs):
        return self.model.transcribe(audio, language=self.language, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.transcribe(*args, **kwargs)


def main(role: str, models_root: str = os.environ.get("MODELS_DIR", "./Models")):
    transmission_sr = 16000
    if role == "server":
        model = WhisperBlock(model_description="medium.en", models_root=models_root)
        with (
            SocketReceiver(sample_rate=transmission_sr, channels=1) as reciever,
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
            SocketStreamer(sample_rate=transmission_sr, channels=1) as streamer,
        ):
            resampler.write_from(audio_stream)
            for segment in resampler:
                streamer.write(segment)
    else:
        raise ValueError("role must be 'server' or 'client'")


if __name__ == "__main__":
    fire.Fire(main)
