from pathlib import Path
from typing import AsyncIterable, Union

from wyoming.audio import AudioChunk

AudioStream = AsyncIterable[AudioChunk]

PathLike = Union[str, Path]
