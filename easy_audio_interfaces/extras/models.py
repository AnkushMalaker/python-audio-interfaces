from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Iterable, List, Optional

import numpy as np
import torch
from numpy.typing import NDArray

from easy_audio_interfaces.audio_interfaces import RechunkingBlock
from easy_audio_interfaces.types.audio import NumpyFrame, NumpySegment
from easy_audio_interfaces.types.common import PathLike


class VoiceGate:
    def __init__(
        self,
        starting_patience: int = 5,
        stopping_patience: int = 20,
        cool_down: int = 20,
        threshold: float = 0.5,
    ) -> None:
        self.starting_patience = starting_patience
        self.stopping_patience = stopping_patience
        self.threshold = threshold
        self._gate: bool = False
        self._frames_in_segment: List[NumpyFrame] = field(default_factory=list)
        self.cooldown = cool_down

        self._starting_patience = starting_patience
        self._stopping_patience = stopping_patience
        self._cooldown = cool_down

    def next(self, probability: float) -> bool:
        # print(self._starting_patience, self._stopping_patience, self._cooldown, probability)
        if not self._gate:
            if self._cooldown > 0:
                # Decrement cooldown counter if it's active
                self._cooldown -= 1
            elif probability > self.threshold:
                self._starting_patience -= 1
                if self._starting_patience <= 0:
                    self._gate = True
                    self._cooldown = self.cooldown  # Reset cooldown on gate opening
            else:
                if self._starting_patience < self.starting_patience:
                    self._starting_patience += 1
        else:
            if probability < self.threshold:
                self._stopping_patience -= 1
                if self._stopping_patience <= 0:
                    self._gate = False
                    # Start cooldown when the gate closes
                    self._cooldown = self.cooldown
            else:
                if self._stopping_patience < self.stopping_patience:
                    self._stopping_patience += 1
        return self._gate

    def iter_segments(
        self, frames: Iterable[NumpyFrame], model: Callable[[torch.Tensor], float]
    ) -> Generator[NumpySegment, None, None]:
        segment = []
        buffer = []
        low_prob_count = 0  # Counter for consecutive low probability frames at segment end

        for frame in frames:
            input_tensor = torch.tensor(frame.normalize(), dtype=torch.float32)
            probability = model(input_tensor)
            if self.next(probability):
                if not self._gate and buffer:
                    # If the gate just opened, prepend buffered frames to the segment
                    segment.extend(buffer)
                    buffer = []
                segment.append(frame)
                low_prob_count = 0  # Reset counter since this is a high probability frame
            else:
                if segment:
                    low_prob_count += 1  # Increment counter for a low probability frame
                    if low_prob_count <= self.stopping_patience:
                        # Add frame to segment if within stopping patience
                        segment.append(frame)
                    if self._stopping_patience <= 0:
                        # Gate closed and there is a segment to yield
                        if low_prob_count > self.stopping_patience:
                            # Discard the low probability frames at the end
                            segment = segment[:-low_prob_count]
                        yield NumpySegment(segment)
                        segment = []
                        low_prob_count = 0
                if self._starting_patience < self.starting_patience:
                    # Buffer frames when voice probability is rising but gate isn't open yet
                    buffer.append(frame)
                else:
                    # Clear the buffer when the voice probability is consistently low
                    buffer = []

        # Yield any remaining segment or buffer when the frames are exhausted
        if segment or buffer:
            yield NumpySegment(segment + buffer)


class SileroVad:
    def __init__(self, sampling_rate: int) -> None:
        # import torch
        # except ImportError:
        #     raise ImportError(
        #         "Please install silero-vad feature to use the SileroVad or install torch."
        #     )

        if sampling_rate not in [16000, 8000]:
            raise ValueError("Sampling rate must be 16000 or 8000")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )
        # (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

        self.WINDOW_SIZE_SAMPLES = (
            512 if sampling_rate == 16000 else 256
        )  # use 256 for 8000 Hz model
        self.model = model
        self.sampling_rate = sampling_rate

    def open(self):
        pass

    def close(self):
        pass

    def __call__(self, frame: NumpyFrame) -> float:
        if len(frame) != self.WINDOW_SIZE_SAMPLES:
            raise ValueError(f"Frame size must be {self.WINDOW_SIZE_SAMPLES} but got {len(frame)}")
        return self.model(frame, self.sampling_rate)

    def voice_segment_iterator(
        self, frames: Iterable[NumpyFrame], voice_gate: VoiceGate = VoiceGate(), **kwargs
    ) -> Generator[NumpySegment, None, None]:
        """Iterate over frames and yield only frames with voice detected
        Args:
        frames (Iterable[NumpyFrame])
        voice_gate (bool | VoiceGate, optional): Defaults to True.

        optional kwargs for :
            starting_patience (int): Defaults to 10.
            stopping_patience (int): Defaults to 10.
            threshold (float): Defaults to 0.5.
        """

        if voice_gate:
            if isinstance(voice_gate, VoiceGate):
                voice_gate = voice_gate
            else:
                voice_gate = VoiceGate(**kwargs)
        yield from voice_gate.iter_segments(
            # self.to_tensor(frames), partial(self.model, sr=self.sampling_rate)
            frames,
            partial(self.model, sr=self.sampling_rate),
        )


class WhisperBlock:
    def __init__(
        self,
        model_description: str = "large-v3",
        language: Optional[str] = None,
        models_root: Optional[PathLike] = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError("Please install stt feature to use the WhisperBlock")
        self.language = language or "en"
        self.model = WhisperModel(
            model_description,
            download_root=str(models_root) if models_root else None,
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


__all__ = ["SileroVad", "WhisperBlock", "VoiceGate"]
