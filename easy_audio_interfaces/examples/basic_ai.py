import asyncio
import logging

from easy_audio_interfaces.audio_interfaces import ResamplingBlock, SocketReceiver
from easy_audio_interfaces.extras.models import SileroVad, WhisperBlock
from easy_audio_interfaces.types.audio import NumpySegment

logging.basicConfig(level=logging.DEBUG)

SEGMENTS_DIR = "./voice-clips-basic-ai/"


async def main_async():
    whisper = WhisperBlock(model_description="large-v3", models_root="./Models")
    model_sampling_rate = whisper.model.feature_extractor.sampling_rate
    silero = SileroVad(sampling_rate=model_sampling_rate)
    async with SocketReceiver(sample_rate=16000) as input_stream:
        resampler = ResamplingBlock(
            original_sample_rate=input_stream.sample_rate, resample_rate=model_sampling_rate
        )
        resampled_stream = resampler.resample(input_stream)
        # Implement RechunkingBlock as an asynchronous iterator if needed
        voice_segments = silero.voice_segment_iterator(resampled_stream)
        async for i, voice_segment in enumerate(voice_segments):
            waveformf32 = voice_segment.normalize().flatten()
            segments, _ = whisper.transcribe(waveformf32, vad_filter=False)
            segments = list(segments)
            if segments:
                for segment in segments:
                    print(segment.text)
                    if segment.text.lower() == "turn on timmy":
                        print("Turning on Timmy")
            # Optional: Save segments or perform additional actions


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
