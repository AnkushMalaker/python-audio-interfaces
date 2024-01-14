import logging

import torch

from easy_audio_interfaces.audio_interfaces import (
    CollectorBlock,
    InputMicStream,
    OutputFileStream,
    RechunkingBlock,
    ResamplingBlock,
)
from easy_audio_interfaces.extras.models import SileroVad, WhisperBlock

logging.basicConfig(level=logging.DEBUG)


def main():
    # Assume you've named your living room tube light timmy.
    "turn on timmy"
    whisper = WhisperBlock(model_description="large-v3", models_root="./Models")
    model_sampling_rate = whisper.model.feature_extractor.sampling_rate
    silero = SileroVad(sampling_rate=model_sampling_rate)
    with (
        InputMicStream() as input_stream,
        ResamplingBlock(
            original_sample_rate=input_stream.sample_rate, resample_rate=model_sampling_rate
        ) as resampler,
        # CollectorBlock(sample_rate=resampler.sample_rate, collect_seconds=5) as collector,
    ):
        # print(model_sampling_rate)
        resampler.write_from(input_stream)
        resampled_stream = resampler.iter()
        rechunked_stream = RechunkingBlock(resampled_stream, chunk_size=512)
        voice_segments = silero.voice_segment_iterator(rechunked_stream)
        for i, segment in enumerate(voice_segments):
            with OutputFileStream(
                f"voice-clips-basic-ai/output_{i}.wav", resampler.sample_rate
            ) as output_stream:
                output_stream.write(segment)

        # collector.write_from(resampler)
        # for waveform in collector:
        #     waveformf32 = waveform.normalize()
        #     segments, _ = whisper.transcribe(waveformf32, vad_filter=False)
        #     segments = list(segments)
        #     print(segments)
        #     if len(segments) > 0:
        #         print("Transcript:")
        #         for segment in segments:
        #             print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


if __name__ == "__main__":
    main()
