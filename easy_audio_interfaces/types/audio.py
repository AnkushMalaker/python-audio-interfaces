from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from easy_audio_interfaces.types.common import PathLike

Sample = np.int16


class NumpyFrame(NDArray[Sample]):
    sample_rate: Optional[int]

    def __new__(cls, input: bytes | NDArray[Sample], sample_rate: Optional[int] = None):
        if isinstance(input, bytes):
            obj = np.frombuffer(input, dtype=np.int16).view(cls)
        else:
            obj = np.asarray(input, dtype=np.int16).view(cls)
        obj.sample_rate = sample_rate
        return obj

    def __array_finalize__(self, obj: NDArray[Any] | None) -> None:
        if obj is None:
            return
        self.sample_rate = getattr(obj, "sample_rate", None)

    @classmethod
    def frombuffer(cls, buffer: bytes) -> "NumpyFrame":
        # Perhaps redundant after expanding scope of what 'input' in __new__ can be
        return np.frombuffer(buffer, dtype=np.int16).view(cls)

    def normalize(self) -> NDArray[np.float32]:
        return self.astype(np.float32) / np.iinfo(self.dtype).max

    def save_waveform_to_file(self, file_path: PathLike):
        raise NotImplementedError


# A segment is the same in terms of data as a frame but is semantically different, as its
# meant to denote a chunk of audio
class NumpySegment(NumpyFrame):
    ...
