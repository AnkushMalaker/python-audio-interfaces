import logging

from easy_audio_interfaces.audio_interfaces import (
    InputMicStream,
    OutputFileStream,
    RechunkingBlock,
    ResamplingBlock,
)
from easy_audio_interfaces.extras.models import SileroVad, WhisperBlock
from easy_audio_interfaces.types.audio import NumpySegment

logging.basicConfig(level=logging.DEBUG)

SEGMENTS_DIR = "./voice-clips-basic-ai/"


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
    ):
        resampler.write_from(input_stream)
        resampled_stream = resampler.iter()
        rechunked_stream = RechunkingBlock(resampled_stream, chunk_size=512)
        voice_segments = silero.voice_segment_iterator(rechunked_stream)
        for i, voice_segment in enumerate(voice_segments):
            waveformf32 = voice_segment.normalize()
            waveformf32 = waveformf32.flatten()
            # with OutputFileStream(
            #     f"./voice-clips-basic-ai/segment_{i}.wav", resampler.sample_rate
            # ) as output_stream:
            #     output_stream.write(NumpySegment(waveformf32))
            segments, _ = whisper.transcribe(waveformf32, vad_filter=False)
            segments = list(segments)
            if len(segments) > 0:
                # print("Transcript:")
                for segment in segments:
                    # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
                    print(segment.text)
                    if segment.text == "turn on timmy":
                        print("Turning on timmy")

                with OutputFileStream(
                    f"./voice-clips-basic-ai/segment_{i}.wav", resampler.sample_rate
                ) as output_stream:
                    output_stream.write(voice_segment)

                with open(f"./voice-clips-basic-ai/segment_{i}.txt", "w") as f:
                    for segment in segments:
                        f.write(segment.text + "\n")


if __name__ == "__main__":
    main()
