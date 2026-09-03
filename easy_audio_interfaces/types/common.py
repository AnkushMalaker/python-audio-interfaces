from pathlib import Path
from typing import AsyncIterable, Union

from .audio import AudioChunk

AudioStream = AsyncIterable[AudioChunk]

PathLike = Union[str, Path]

__all__ = ["AudioStream", "PathLike"]
