import sounddevice as sd
import numpy as np
from typing import Optional


def resolve_input_device(preferred: Optional[str] = None):
    # Force system default input device.
    return None


def resolve_input_device_with_rate(preferred: Optional[str] = None):
    device = None
    try:
        info = sd.query_devices(device=device, kind="input")
    except Exception:
        info = sd.query_devices(kind="input")
    default_rate = int(float(info.get("default_samplerate", 16000)))
    return device, default_rate


def _resample_int16_mono(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return audio
    if audio.size == 0:
        return audio

    src_len = audio.shape[0]
    dst_len = int(round(src_len * (dst_rate / float(src_rate))))
    if dst_len <= 1:
        return audio[:1]

    x_src = np.linspace(0.0, 1.0, src_len, endpoint=False)
    x_dst = np.linspace(0.0, 1.0, dst_len, endpoint=False)
    resampled = np.interp(x_dst, x_src, audio.astype(np.float32))
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def record_audio(
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
    input_device: Optional[str] = None
) -> bytes:
    """
    Records audio from the default microphone and returns raw PCM bytes.
    """

    device, capture_rate = resolve_input_device_with_rate(input_device)
    frames = int(duration_sec * capture_rate)

    recording = sd.rec(
        frames,
        samplerate=capture_rate,
        channels=1,
        dtype="int16",
        device=device
    )

    sd.wait()
    mono = recording.reshape(-1)
    mono_16k = _resample_int16_mono(mono, capture_rate, sample_rate)
    return mono_16k.tobytes()
