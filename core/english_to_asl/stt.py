from typing import Tuple


def speech_to_text(audio: bytes) -> Tuple[str, float]:
    """
    Convert raw audio bytes into English text.

    Args:
        audio: Raw audio bytes (PCM/WAV data)

    Returns:
        (text, confidence)
    """
    raise NotImplementedError("Speech-to-text backend not implemented yet")