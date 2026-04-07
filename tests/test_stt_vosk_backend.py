import pytest
from core.english_to_asl.stt_vosk import VoskSTT


def test_vosk_backend_loads():
    try:
        stt = VoskSTT()
    except Exception as exc:
        pytest.skip(f"Local Vosk model unavailable: {exc}")
    assert stt is not None
