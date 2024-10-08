import logging
import time
from enum import Enum
from typing import Optional

import fire

from easy_audio_interfaces.audio_interfaces import (
    InputMicStream,
    OutputFileStream,
    OutputSpeakerStream,
    ResamplingBlock,
)
from easy_audio_interfaces.types.common import PathLike

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def mic_to_file(
    mic_index: Optional[int] = None,
    filepath: PathLike = "./recorded.wav",
):
    output_file_sr = 16000
    with (
        InputMicStream(device_index=mic_index) as mic_stream,
        OutputFileStream(filepath, sample_rate=output_file_sr) as output_file,
        ResamplingBlock(
            resample_rate=output_file_sr, conversion_method="sinc_best"
        ) as resampling_block,
    ):
        resampling_block.write_from(mic_stream)
        output_file.write_from(resampling_block)
        time.sleep(1)


def mic_to_speaker(
    mic_index: Optional[int] = None,
    speaker_index: Optional[int] = None,
):
    with (
        InputMicStream(mic_index) as mic_stream,
        OutputSpeakerStream(speaker_index) as speaker_stream,
    ):
        with ResamplingBlock(
            original_sample_rate=mic_stream.sample_rate,
            resample_rate=8000,
            conversion_method="sinc_best",
        ) as resampling_block1, ResamplingBlock(
            original_sample_rate=8000,
            resample_rate=speaker_stream.sample_rate,
            conversion_method="sinc_best",
        ) as resampling_block2:
            resampling_block1.write_from(mic_stream)
            resampling_block2.write_from(resampling_block1)
            speaker_stream.write_from(resampling_block2)
            time.sleep(10)


# FIXME: Look into fire.Fire and allow different commands to be run from the command line


def main():
    fire.Fire(mic_to_speaker)


if __name__ == "__main__":
    fire.Fire(main)
