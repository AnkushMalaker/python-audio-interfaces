# Deprecated
# from functools import partial
# from typing import Any, AsyncGenerator, AsyncIterable, Callable, List, Optional, Tuple

# import numpy as np
# import torch
# from numpy.typing import NDArray

# from easy_audio_interfaces.types.audio import NumpyFrame, NumpySegment
# from easy_audio_interfaces.types.common import PathLike


# class VoiceGate:
#     def __init__(
#         self,
#         starting_patience: int = 5,
#         stopping_patience: int = 20,
#         cool_down: int = 20,
#         threshold: float = 0.5,
#     ) -> None:
#         """
#         Args:
#             starting_patience: Number of frames to wait before starting to record.
#             stopping_patience: Number of frames to wait before stopping to record.
#             cool_down: Number of frames to wait before allowing the gate to open again.
#             threshold: Probability threshold to determine if the gate should open.
#         """
#         self.starting_patience = starting_patience
#         self.stopping_patience = stopping_patience
#         self.threshold = threshold
#         self._gate: bool = False
#         self.cooldown = cool_down

#         self._starting_patience = starting_patience
#         self._stopping_patience = stopping_patience
#         self._cooldown = 0  # Start without cooldown

#     def next(self, probability: float) -> Tuple[bool, bool, bool]:
#         """
#         Update the gate state based on the given probability.

#         Returns:
#             gate_open: whether the gate is open after processing this frame
#             gate_just_opened: whether the gate has just opened
#             gate_just_closed: whether the gate has just closed
#         """
#         gate_was_open = self._gate

#         if not self._gate:
#             if self._cooldown > 0:
#                 self._cooldown -= 1
#             elif probability > self.threshold:
#                 self._starting_patience -= 1
#                 if self._starting_patience <= 0:
#                     self._gate = True
#                     self._cooldown = self.cooldown
#                     self._starting_patience = self.starting_patience  # Reset for next time
#                     self._stopping_patience = self.stopping_patience  # Reset stopping patience
#             else:
#                 if self._starting_patience < self.starting_patience:
#                     self._starting_patience += 1
#         else:
#             if probability < self.threshold:
#                 self._stopping_patience -= 1
#                 if self._stopping_patience <= 0:
#                     self._gate = False
#                     self._cooldown = self.cooldown
#                     self._starting_patience = self.starting_patience  # Reset starting patience
#                     self._stopping_patience = self.stopping_patience  # Reset for next time
#             else:
#                 if self._stopping_patience < self.stopping_patience:
#                     self._stopping_patience += 1

#         gate_just_opened = self._gate and not gate_was_open
#         gate_just_closed = not self._gate and gate_was_open

#         return self._gate, gate_just_opened, gate_just_closed

#     async def iter_segments(
#         self, frames: AsyncIterable[NumpyFrame], model: Callable[[torch.Tensor], float]
#     ) -> AsyncGenerator[NumpySegment, None]:
#         segment: List[NumpyFrame] = []
#         buffer: List[NumpyFrame] = []
#         frames_since_gate_closed = None

#         with torch.inference_mode():
#             async for frame in frames:
#                 input_tensor = torch.tensor(frame.normalize(), dtype=torch.float32)
#                 probability = model(input_tensor)
#                 gate_open, gate_just_opened, gate_just_closed = self.next(probability)

#                 if gate_open:
#                     if gate_just_opened:
#                         # Include buffered frames collected during starting patience
#                         segment = buffer.copy()
#                         buffer = []
#                         frames_since_gate_closed = None
#                     segment.append(frame)
#                 else:
#                     if segment:
#                         if frames_since_gate_closed is None:
#                             frames_since_gate_closed = 1
#                         else:
#                             frames_since_gate_closed += 1

#                         if frames_since_gate_closed <= self.stopping_patience:
#                             segment.append(frame)
#                         else:
#                             # End of segment
#                             yield NumpySegment(np.concatenate(segment))
#                             segment = []
#                             frames_since_gate_closed = None
#                     else:
#                         # Buffer frames while waiting for gate to open
#                         buffer.append(frame)
#                         if len(buffer) > self.starting_patience:
#                             buffer.pop(0)

#             # Yield any remaining segment
#             if segment:
#                 yield NumpySegment(np.concatenate(segment))

#     @classmethod
#     def init_from_seconds(
#         cls,
#         sample_rate: int,
#         starting_patience_seconds: float,
#         stopping_patience_seconds: float,
#         cool_down_seconds: float,
#         threshold: float,
#     ) -> "VoiceGate":
#         return cls(
#             starting_patience=int(starting_patience_seconds * sample_rate),
#             stopping_patience=int(stopping_patience_seconds * sample_rate),
#             cool_down=int(cool_down_seconds * sample_rate),
#             threshold=threshold,
#         )


# class SileroVad:
#     def __init__(self, sampling_rate: int) -> None:
#         try:
#             import torch  # type: ignore
#         except ImportError:
#             raise ImportError(
#                 "Please install silero-vad feature to use the SileroVad or install torch."
#             )

#         if sampling_rate not in [16000, 8000]:
#             raise ValueError("Sampling rate must be 16000 or 8000")
#         model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")

#         self.WINDOW_SIZE_SAMPLES = 512 if sampling_rate == 16000 else 256
#         self.model = model
#         self.sampling_rate = sampling_rate

#     def open(self):
#         pass

#     def close(self):
#         pass

#     def __call__(self, frame: NumpyFrame | torch.Tensor) -> float:
#         if isinstance(frame, np.ndarray):
#             frame = torch.tensor(frame.normalize(), dtype=torch.float32)
#         if len(frame) != self.WINDOW_SIZE_SAMPLES:
#             raise ValueError(f"Frame size must be {self.WINDOW_SIZE_SAMPLES} but got {len(frame)}")
#         return self.model(frame, self.sampling_rate)

#     async def iter_segments(
#         self, frames: AsyncIterable[NumpyFrame], voice_gate: VoiceGate
#     ) -> AsyncGenerator[NumpySegment, None]:
#         async for segment in voice_gate.iter_segments(
#             frames,
#             partial(self.model, sr=self.sampling_rate),
#         ):
#             yield segment


# class WhisperBlock:
#     def __init__(
#         self,
#         model_description: str = "large-v3",
#         language: Optional[str] = None,
#         models_root: Optional[PathLike] = None,
#         compute_type: str = "default",
#         num_workers: int = 1,
#     ) -> None:
#         try:
#             from faster_whisper import WhisperModel  # type: ignore
#         except ImportError:
#             raise ImportError("Please install stt feature to use the WhisperBlock")
#         self.language = language or "en"
#         self.model = WhisperModel(
#             model_description,
#             download_root=str(models_root) if models_root else None,
#             compute_type=compute_type,
#             num_workers=num_workers,
#         )
#         self.feature_extractor = self.model.feature_extractor

#     def open(self):
#         pass

#     def close(self):
#         pass

#     def transcribe(self, audio: NDArray, **kwargs):
#         return self.model.transcribe(audio, language=self.language, **kwargs)

#     def __call__(self, *args: Any, **kwargs: Any) -> Any:
#         return self.transcribe(*args, **kwargs)


# __all__ = ["SileroVad", "WhisperBlock", "VoiceGate"]
