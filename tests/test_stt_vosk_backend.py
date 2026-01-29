import pytest
from core.english_to_asl.stt_vosk import VoskSTT


def test_vosk_backend_loads():
    stt = VoskSTT()
    assert stt is not None