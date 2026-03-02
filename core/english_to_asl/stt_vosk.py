from typing import Tuple
import json
from vosk import Model, KaldiRecognizer


MODEL_PATH = "models/vosk-en"


class VoskSTT:
    def __init__(self, sample_rate: int = 16000):
        self.model = Model(MODEL_PATH)
        self.sample_rate = sample_rate

    def speech_to_text(self, audio: bytes) -> Tuple[str, float]:
        """
        Convert raw PCM audio bytes into English text using Vosk.
        """
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        recognizer.AcceptWaveform(audio)
        result = json.loads(recognizer.FinalResult())

        text = result.get("text", "").strip()
        confidence = 1.0 if text else 0.0  # placeholder confidence

        return text, confidence
