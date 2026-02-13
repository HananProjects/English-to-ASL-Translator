import sounddevice as sd
import numpy as np
from typing import Optional
import os


def _system_default_input_index() -> Optional[int]:
    """Return OS default input device index when available."""
    try:
        default_pair = sd.default.device
        if isinstance(default_pair, (list, tuple)) and len(default_pair) >= 1:
            idx = default_pair[0]
            if idx is not None and int(idx) >= 0:
                return int(idx)
    except Exception:
        pass
    return None


def resolve_input_device(preferred: Optional[str] = None):
    """
    Resolve input device from:
    1) explicit `preferred`
    2) env override `ASL_INPUT_DEVICE`
    3) OS default input device
    """
    env_preferred = os.getenv("ASL_INPUT_DEVICE")
    choice = preferred or env_preferred

    if choice is None or str(choice).strip() == "":
        return _system_default_input_index()

    choice = str(choice).strip()

    # Allow direct index selection, e.g. ASL_INPUT_DEVICE=3
    if choice.isdigit():
        return int(choice)

    needle = choice.lower()
    for idx, dev in enumerate(sd.query_devices()):
        name = str(dev.get("name", ""))
        if dev.get("max_input_channels", 0) > 0 and needle in name.lower():
            return idx

    # Fallback to OS default if requested name is not found.
    return _system_default_input_index()


def resolve_input_device_with_rate(preferred: Optional[str] = None):
    device = resolve_input_device(preferred)
    try:
        info = sd.query_devices(device=device, kind="input")
    except Exception:
        device = _system_default_input_index()
        try:
            info = sd.query_devices(device=device, kind="input")
        except Exception:
            device = None
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
    Records audio from the selected/default microphone and returns raw PCM bytes.
    """

    device, capture_rate = resolve_input_device_with_rate(input_device)
    try:
        dev_info = sd.query_devices(device=device, kind="input")
        dev_name = dev_info.get("name", "default")
    except Exception:
        dev_name = "default"
    print(f"[Mic] using device={device} name={dev_name} rate={capture_rate}")

    frames = int(duration_sec * capture_rate)
    recording = sd.rec(
        frames,
        samplerate=capture_rate,
        channels=1,
        dtype="int16",
        device=device,
    )

    sd.wait()
    mono = recording.reshape(-1)
    mono_16k = _resample_int16_mono(mono, capture_rate, sample_rate)
    return mono_16k.tobytes()
