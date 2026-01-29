import soundfile as sf
from typing import Tuple


def load_wav_as_pcm(path: str, target_sr: int = 16000) -> Tuple[bytes, int]:
    """
    Load a WAV file and return raw PCM bytes and sample rate.
    Vosk expects 16kHz mono PCM.
    """
    data, sr = sf.read(path, dtype="int16")

    if len(data.shape) > 1:
        # convert stereo to mono
        data = data.mean(axis=1).astype("int16")

    if sr != target_sr:
        raise ValueError(f"Expected {target_sr} Hz audio, got {sr} Hz")

    return data.tobytes(), sr