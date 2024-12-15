import asyncio
import logging
from functools import partial
from typing import Optional

import numpy as np
from openwakeword.model import Model

from easy_audio_interfaces.network import SocketServer

logging.basicConfig(level=logging.INFO)


class WakewordDetector:
    def __init__(self, model_path=""):
        # Load pre-trained openwakeword models
        if model_path:
            self.owwModel = Model(wakeword_models=[model_path])
        else:
            self.owwModel = Model()

        self.n_models = len(self.owwModel.models.keys())
        self.print_header()

    def print_header(self):
        print("\n\n")
        print("#" * 100)
        print("Listening for wakewords...")
        print("#" * 100)
        print("\n" * (self.n_models * 3))

    def process_audio(self, audio_chunk):
        # Convert audio chunk to numpy array
        audio = np.frombuffer(audio_chunk, dtype=np.int16)

        # Feed to openWakeWord model
        prediction = self.owwModel.predict(audio)

        # Format and print results
        n_spaces = 16
        output_string_header = """
            Model Name         | Score | Wakeword Status
            --------------------------------------
            """

        for mdl in self.owwModel.prediction_buffer.keys():
            scores = list(self.owwModel.prediction_buffer[mdl])
            curr_score = format(scores[-1], ".20f").replace("-", "")

            output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
            """

        # Print results table
        print("\033[F" * (4 * self.n_models + 1))
        print(output_string_header, "                             ", end="\r")


async def serve():
    server = SocketServer(host="0.0.0.0", port=4369)
    detector = WakewordDetector()

    async with server as server:
        async for chunk in server:
            detector.process_audio(chunk)


if __name__ == "__main__":
    asyncio.run(serve())
