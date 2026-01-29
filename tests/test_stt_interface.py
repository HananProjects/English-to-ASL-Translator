import pytest
from core.english_to_asl.stt import speech_to_text


def test_stt_not_implemented():
    with pytest.raises(NotImplementedError):
        speech_to_text(b"fake_audio_bytes")