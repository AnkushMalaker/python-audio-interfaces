from typing import Callable, Union

import numpy as np
import torch
import torchaudio
from numpy.typing import NDArray

from easy_audio_interfaces.types.audio import NumpyFrame


def int2float(sound: NDArray[np.int16]) -> NDArray[np.float32]:
    abs_max = np.abs(sound).max()
    float_sound = sound.astype("float32")
    if abs_max > 0:
        float_sound *= 1 / abs_max
    float_sound = float_sound.squeeze()  # depends on the use case
    return float_sound


def frame_to_spectrogram(
    frame: Union[NumpyFrame, torch.Tensor],
    pad: int = 0,
    n_ftt: int = 400,
    normalized: bool = False,
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """Returns:
    Tensor: Dimension (..., freq, time), freq is n_fft // 2 + 1 and n_fft is the number of Fourier bins, and time is the number of window hops (n_frame).
    """
    # Too heavy for types/audio.py
    spectrogram = torchaudio.functional.spectrogram(
        torch.FloatTensor(frame) if isinstance(frame, np.ndarray) else frame,
        pad=pad,
        window=window_fn(n_ftt).to(device),
        n_fft=n_ftt,
        power=2.0,
        normalized=normalized,
        hop_length=n_ftt // 2,
        win_length=n_ftt,
    )
    return spectrogram


def frame_to_stft(
    frame: NumpyFrame,
    n_ftt: int = 400,
    normalized: bool = True,
) -> torch.Tensor:
    """Returns:
    Tensor: Dimension (..., freq, time), freq is n_fft // 2 + 1 and n_fft is the number of Fourier bins, and time is the number of window hops (n_frame).
    """

    # Too heavy for types/audio.py
    stft = torch.stft(
        torch.from_numpy(frame),
        n_fft=n_ftt,
        normalized=normalized,
        return_complex=True,
    )
    stft = torch.view_as_real(stft)
    return stft


if __name__ == "__main__":
    frame = NumpyFrame(np.random.rand(16000))
    spec = frame_to_spectrogram(frame)
    stft = frame_to_stft(frame)
    print(frame.shape)
    print(spec.shape)  # 201,81
    print(stft.shape)  # 201, 161
