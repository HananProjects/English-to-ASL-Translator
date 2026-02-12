from PySide6.QtCore import QObject, Signal
from core.engine import english_to_asl
from core.mic_utils import record_audio


class TranslationWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def run(self):
        try:
            print("[Record] capture start")
            audio = record_audio(duration_sec=3.0)
            print(f"[Record] captured bytes={len(audio)}")
            result = english_to_asl(audio=audio)
            print(
                "[Record] stt text=",
                repr(result.source_text),
                "tokens=",
                result.asl_tokens,
                "error=",
                result.error,
            )

            self.finished.emit({
                "tokens": result.asl_tokens,
                "confidence": result.confidence,
                "latency": result.latency_ms,
                "text": result.source_text,
                "error": result.error,
            })

        except Exception as e:
            print(f"[Record] exception: {e}")
            self.error.emit(str(e))
