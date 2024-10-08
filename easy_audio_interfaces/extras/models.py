from functools import partial
from typing import Any, AsyncGenerator, AsyncIterable, Callable, List, Optional

import numpy as np
import torch
from numpy.typing import NDArray

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
        """
        Args:
            starting_patience: Number of frames to wait before starting to record.
            stopping_patience: Number of frames to wait before stopping to record.
            cool_down: Number of frames to wait before allowing the gate to open again.
            threshold: Probability threshold to determine if the gate should open.
        """
        self.starting_patience = starting_patience
        self.stopping_patience = stopping_patience
        self.threshold = threshold
        self._gate: bool = False
        self._frames_in_segment: List[NumpyFrame] = []
        self.cooldown = cool_down

        self._starting_patience = starting_patience
        self._stopping_patience = stopping_patience
        self._cooldown = cool_down

    def next(self, probability: float) -> bool:
        if not self._gate:
            if self._cooldown > 0:
                self._cooldown -= 1
            elif probability > self.threshold:
                self._starting_patience -= 1
                if self._starting_patience <= 0:
                    self._gate = True
                    self._cooldown = self.cooldown
            else:
                if self._starting_patience < self.starting_patience:
                    self._starting_patience += 1
        else:
            if probability < self.threshold:
                self._stopping_patience -= 1
                if self._stopping_patience <= 0:
                    self._gate = False
                    self._cooldown = self.cooldown
            else:
                if self._stopping_patience < self.stopping_patience:
                    self._stopping_patience += 1
        return self._gate

    @torch.inference_mode()
    async def iter_segments(
        self, frames: AsyncIterable[NumpyFrame], model: Callable[[torch.Tensor], float]
    ) -> AsyncGenerator[NumpySegment, None]:
        segment = []
        buffer = []
        low_prob_count = 0

        with torch.inference_mode():
            async for frame in frames:
                input_tensor = torch.tensor(frame.normalize(), dtype=torch.float32)
                probability = model(input_tensor)
                if self.next(probability):
                    if not self._gate and buffer:
                        segment.extend(buffer)
                        buffer = []
                    segment.append(frame)
                    low_prob_count = 0
                else:
                    if segment:
                        low_prob_count += 1
                        if low_prob_count <= self.stopping_patience:
                            segment.append(frame)
                        else:
                            # Yield the segment when low_prob_count exceeds stopping_patience
                            segment = segment[: -self.stopping_patience]
                            yield NumpySegment(np.concatenate(segment))
                            segment = []
                            low_prob_count = 0
                    if self._starting_patience < self.starting_patience:
                        buffer.append(frame)
                    else:
                        buffer = []

            if segment:
                yield NumpySegment(np.concatenate(segment))
            elif buffer:
                yield NumpySegment(np.concatenate(buffer))


class SileroVad:
    def __init__(self, sampling_rate: int) -> None:
        try:
            import torch  # type: ignore
        except ImportError:
            raise ImportError(
                "Please install silero-vad feature to use the SileroVad or install torch."
            )

        if sampling_rate not in [16000, 8000]:
            raise ValueError("Sampling rate must be 16000 or 8000")
        model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")

        self.WINDOW_SIZE_SAMPLES = 512 if sampling_rate == 16000 else 256
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

    async def iter_segments(
        self, frames: AsyncIterable[NumpyFrame], voice_gate: VoiceGate
    ) -> AsyncGenerator[NumpySegment, None]:
        async for segment in voice_gate.iter_segments(
            frames,
            partial(self.model, sr=self.sampling_rate),
        ):
            yield segment


class WhisperBlock:
    def __init__(
        self,
        model_description: str = "large-v3",
        language: Optional[str] = None,
        models_root: Optional[PathLike] = None,
    ) -> None:
        try:
            from faster_whisper import WhisperModel  # type: ignore
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
