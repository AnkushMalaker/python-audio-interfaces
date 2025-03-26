from pathlib import Path
from typing import AsyncIterable, Union

from pydub import AudioSegment

AudioStream = AsyncIterable[AudioSegment]

PathLike = Union[str, Path]
