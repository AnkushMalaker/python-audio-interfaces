import tkinter as tk
import wave
from functools import partial
from typing import Any, Optional, Protocol, Tuple

import fire
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
from PIL import ImageTk as PILImageTk
from scipy.signal import spectrogram

from easy_audio_interfaces.audio_interfaces import (
    AudioSource,
    InputFileStream,
    InputMicStream,
)
from easy_audio_interfaces.types.audio import NumpyFrame
from easy_audio_interfaces.types.common import PathLike


class AudioTransformer(Protocol):
    def __call__(self, x: np.ndarray, sample_rate: float) -> np.ndarray:
        ...


class DataCollectionGUI:
    """Shows the distance of mouse position from the edges of the application."""

    _audio: NumpyFrame

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Data Collection GUI")
        self.root.geometry("600x400")
        self.root.bind("<Motion>", self.second_func)

        self.transform: AudioTransformer = lambda x, sample_rate: partial(
            spectrogram, return_onesided=False
        )(x, sample_rate)[2]

    @property
    def audio_array(self):
        if self._audio is None:
            raise ValueError("No audio loaded")

        return self._audio

    @property
    def spectrogram(self) -> np.ndarray:
        return self.transform(self.audio_array, self.sample_rate)[2]

    @property
    def sample_rate(self) -> int:
        assert self._audio.sample_rate is not None, "Audio must have sample rate."
        return self._audio.sample_rate

    def second_func(self, event):
        """Update the label with the mouse position."""
        print(f"({event.x}, {event.y})")

    def show_spectrogram(self):
        # spectrogram = self.spectrogram
        # spectrogram = np.log(spectrogram + 1e-9)
        # spectrogram = spectrogram / np.max(spectrogram)
        # spectrogram = np.uint8(spectrogram * 255)
        # spectrogram = PILImage.fromarray(spectrogram)
        # spectrogram.save("spectrogram.png")
        # spectrogram_image = PILImageTk.PhotoImage(spectrogram)

        # pxx, freqs, bins, im = plt.specgram(self._audio, Fs=self.sample_rate)
        # canvas = plt.get_current_fig_manager().canvas
        # if canvas is not None:
        #     plt.axis("off")
        #     plt.tight_layout(pad=0)
        #     canvas.draw()
        #     spectrogram_image = PILImageTk.PhotoImage(
        #         PILImage.frombytes("RGB", canvas.get_width_height(), canvas.buffer_rgba())
        #     )
        # else:
        #     raise ValueError("No canvas found")
        # spectrogram_image = PILImageTk.PhotoImage(PILImage.fromarray(pxx))

        # spectrogram_label = tk.Label(self.root, image=spectrogram_image)
        # spectrogram_label.image = spectrogram_image  # type: ignore
        # spectrogram_label.pack()
        pass

    def start(self):
        self.show_spectrogram()
        self.root.mainloop()

    def load_audio_from_stream(self, audio_stream: AudioSource):
        frames = [audio_frame for audio_frame in audio_stream]
        frames = np.concatenate(frames)
        self._audio = NumpyFrame(frames, sample_rate=audio_stream.sample_rate)

    def load_audio_from_file(self, audio_file: PathLike):
        with wave.open(str(audio_file), "rb") as audio:
            frames = audio.readframes(audio.getnframes())
            frames = np.frombuffer(frames, dtype=np.int16)
            self._audio = NumpyFrame(frames, sample_rate=audio.getframerate())


# FIXME: input_file should have None default, doing this for testing right now
def main(input_file: Optional[PathLike]):
    input_audio_stream = (
        InputFileStream(input_file).start() if input_file is not None else InputMicStream()
    )
    gui = DataCollectionGUI()
    gui.load_audio_from_stream(input_audio_stream)
    gui.start()


if __name__ == "__main__":
    fire.Fire(main)
