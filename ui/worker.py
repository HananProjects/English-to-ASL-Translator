from PySide6.QtCore import QObject, Signal
from core.engine import english_to_asl
from core.mic_utils import record_audio


class TranslationWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def run(self):
        try:
            audio = record_audio(duration_sec=3.0)
            result = english_to_asl(audio=audio)

            if result.error:
                self.error.emit(result.error)
                return

            self.finished.emit({
                "tokens": result.asl_tokens,
                "confidence": result.confidence,
                "latency": result.latency_ms
            })

        except Exception as e:
            self.error.emit(str(e))