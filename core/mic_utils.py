import sounddevice as sd
import numpy as np


def record_audio(duration_sec: float = 3.0, sample_rate: int = 16000) -> bytes:
    """
    Records audio from the default microphone and returns raw PCM bytes.
    """

    frames = int(duration_sec * sample_rate)

    recording = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=1,
        dtype="int16"
    )

    sd.wait()

    return recording.tobytes()