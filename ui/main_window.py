from core.engine import english_to_asl, _get_stt_backend
from core.mic_utils import record_audio
from core.audio.vosk_listener import VoskListener
import threading
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from core.sequencing.sign_sequencer import sequence_signs
from ui.widgets.animation_view import ASLAnimationView
from ui.worker_camera import CameraWorker
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton


class MainWindow(QWidget):
    speech_text_received = Signal(str)
    record_result_received = Signal(dict)
    record_error_received = Signal(str)
    stt_warmup_received = Signal(bool)

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.vosk_init_in_progress = False

        self.setWindowTitle("English to ASL Translator")
        self.setMinimumSize(1000, 700)
        self.resize(1280, 820)

        layout = QVBoxLayout()

        self.animation_view = ASLAnimationView()
        self.animation_view.setMinimumHeight(620)
        # Keep avatar stationary by default in English->ASL mode.
        self.animation_view.disable_live_pose()
        layout.addWidget(self.animation_view, 5)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-size: 18px;")

        self.tokens_label = QLabel("ASL Output:")
        self.tokens_label.setStyleSheet("font-size: 22px;")
        self.clips_label = QLabel("Clips: (none)")
        self.clips_label.setStyleSheet("font-size: 16px;")

        self.record_button = QPushButton("Record")
        self.record_button.setStyleSheet("font-size: 20px; height: 60px;")
        self.record_button.clicked.connect(self.on_record_clicked)
        self.record_button.setEnabled(False)

        layout.addWidget(self.status_label)
        layout.addWidget(self.tokens_label)
        layout.addWidget(self.clips_label)
        layout.addWidget(self.record_button)

        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)
        self.setLayout(layout)

        self.speech_text_received.connect(self.on_speech)
        self.record_result_received.connect(self.on_translation_finished)
        self.record_error_received.connect(self.on_translation_error)
        self.stt_warmup_received.connect(self._on_stt_warmup_complete)

        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.pose_ready.connect(self.animation_view.set_live_pose)
        self.camera_thread.start()

        self.vosk = None
        self.status_label.setText("Status: Loading speech model...")
        threading.Thread(target=self._warmup_stt_backend, daemon=True).start()

    def on_record_clicked(self):
        self.status_label.setText("Status: Recording...")
        self.tokens_label.setText("ASL Output:")
        self.record_button.setEnabled(False)

        if getattr(self, "vosk", None) is not None:
            self.vosk.stop()

        threading.Thread(target=self._run_record_job, daemon=True).start()

    def on_translation_finished(self, data):
        self.record_button.setEnabled(True)
        tokens = data["tokens"]
        heard_text = data.get("text", "")
        error = data.get("error")

        self.tokens_label.setText(f"ASL Output: {' '.join(tokens)}")
        if error:
            self.status_label.setText(
                f"Heard: {heard_text or '(none)'} | Error: {error} | "
                f"Latency: {data['latency']} ms"
            )
        else:
            self.status_label.setText(
                f"Heard: {heard_text or '(none)'} | "
                f"Confidence: {data['confidence']:.2f} | "
                f"Latency: {data['latency']} ms"
            )

        if not tokens:
            self.clips_label.setText("Clips: (none)")
            return

        sequence = sequence_signs(tokens)
        self._set_clips_label(sequence)
        self.animation_view.disable_live_pose()
        self.animation_view.play(sequence)

    def on_translation_error(self, message):
        print(f"[UI] translation error: {message}")
        self.record_button.setEnabled(True)
        self.status_label.setText(f"Error: {message}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        if getattr(self, "vosk", None) is not None:
            self.vosk.stop()

        if hasattr(self, "camera_thread"):
            self.camera_thread.requestInterruption()
            self.camera_thread.quit()
            self.camera_thread.wait()
        event.accept()

    def on_speech(self, text: str):
        print("Heard:", text)

        result = english_to_asl(text=text)
        tokens = result.asl_tokens
        if not tokens:
            if result.error:
                self.status_label.setText(f"Status: {result.error}")
            return

        self.tokens_label.setText(f"ASL Output: {' '.join(tokens)}")
        self.status_label.setText(
            f"Status: Live Speech | Confidence: {result.confidence:.2f}"
        )

        sequence = sequence_signs(tokens)
        self._set_clips_label(sequence)
        self.animation_view.disable_live_pose()
        self.animation_view.play(sequence)

    def _start_vosk_async(self):
        if self.vosk is not None:
            if not self.vosk.running:
                self.vosk.start()
            return
        if self.vosk_init_in_progress:
            return
        self.vosk_init_in_progress = True
        threading.Thread(target=self._init_vosk_listener, daemon=True).start()

    def _run_record_job(self):
        try:
            print("[Record] capture start")
            audio = record_audio(duration_sec=4.0)
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
            self.record_result_received.emit({
                "tokens": result.asl_tokens,
                "confidence": result.confidence,
                "latency": result.latency_ms,
                "text": result.source_text,
                "error": result.error,
            })
        except Exception as e:
            self.record_error_received.emit(str(e))

    def _init_vosk_listener(self):
        try:
            if self.vosk is not None:
                return
            listener = VoskListener(
                model_path="models/vosk-en",
                on_text=self.speech_text_received.emit,
            )
            listener.start()
            self.vosk = listener
        except Exception as e:
            self.vosk = None
            print(f"Vosk init error: {e}")
        finally:
            self.vosk_init_in_progress = False

    def _warmup_stt_backend(self):
        try:
            _get_stt_backend()
            print("[STT] backend warmed")
            self.stt_warmup_received.emit(True)
        except Exception as e:
            print(f"[STT] warmup failed: {e}")
            self.stt_warmup_received.emit(False)

    def _on_stt_warmup_complete(self, ok: bool):
        self.record_button.setEnabled(True)
        if ok:
            self.status_label.setText("Status: Idle")
        else:
            self.status_label.setText("Status: STT warmup failed")

    def _set_clips_label(self, sequence):
        if not sequence:
            self.clips_label.setText("Clips: (none)")
            return
        clip_parts = [f"{event.token}:{event.clip}.json" for event in sequence]
        self.clips_label.setText("Clips: " + " | ".join(clip_parts))
