from core.engine import english_to_asl, asl_to_english, _get_stt_backend
from core.mic_utils import record_audio
from core.audio.vosk_listener import VoskListener
import threading
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from core.sequencing.sign_sequencer import sequence_signs
from ui.widgets.animation_view import ASLAnimationView
from ui.worker_camera import CameraWorker
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QLabel,
    QPushButton,
)


class MainWindow(QWidget):
    speech_text_received = Signal(str)
    record_result_received = Signal(dict)
    record_error_received = Signal(str)
    stt_warmup_received = Signal(bool)
    camera_sequence_received = Signal(list)
    camera_token_received = Signal(str, float)

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.vosk_init_in_progress = False

        self.setWindowTitle("English <-> ASL Translator")
        self.setMinimumSize(1000, 700)
        self.resize(1280, 820)

        layout = QVBoxLayout()

        self.mode = "english_to_asl"

        mode_row = QHBoxLayout()
        self.english_mode_button = QPushButton("English -> ASL")
        self.reverse_mode_button = QPushButton("ASL -> English")
        self.english_mode_button.clicked.connect(
            lambda: self.set_mode("english_to_asl")
        )
        self.reverse_mode_button.clicked.connect(
            lambda: self.set_mode("asl_to_english")
        )
        mode_row.addWidget(self.english_mode_button)
        mode_row.addWidget(self.reverse_mode_button)
        layout.addLayout(mode_row)

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._build_english_to_asl_page())
        self.mode_stack.addWidget(self._build_asl_to_english_page())
        layout.addWidget(self.mode_stack, 1)

        layout.setContentsMargins(24, 16, 24, 16)
        layout.setSpacing(12)
        self.setLayout(layout)
        self._refresh_mode_buttons()

        self.speech_text_received.connect(self.on_speech)
        self.record_result_received.connect(self.on_translation_finished)
        self.record_error_received.connect(self.on_translation_error)
        self.stt_warmup_received.connect(self._on_stt_warmup_complete)
        self.camera_sequence_received.connect(self.on_camera_sequence)
        self.camera_token_received.connect(self.on_camera_token)

        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.pose_ready.connect(self.on_camera_pose)
        self.camera_worker.token_ready.connect(self.camera_token_received.emit)
        self.camera_worker.sequence_ready.connect(self.camera_sequence_received.emit)
        self.camera_thread.start()

        self.vosk = None
        self.english_status_label.setText("Status: Loading speech model...")
        threading.Thread(target=self._warmup_stt_backend, daemon=True).start()

    def on_record_clicked(self):
        self.english_status_label.setText("Status: Recording...")
        self.english_tokens_label.setText("ASL Output:")
        self.record_button.setEnabled(False)

        if getattr(self, "vosk", None) is not None:
            self.vosk.stop()

        threading.Thread(target=self._run_record_job, daemon=True).start()

    def on_translation_finished(self, data):
        self.record_button.setEnabled(True)
        tokens = data["tokens"]
        heard_text = data.get("text", "")
        error = data.get("error")

        self.english_tokens_label.setText(f"ASL Output: {' '.join(tokens)}")
        if error:
            self.english_status_label.setText(
                f"Heard: {heard_text or '(none)'} | Error: {error} | "
                f"Latency: {data['latency']} ms"
            )
        else:
            self.english_status_label.setText(
                f"Heard: {heard_text or '(none)'} | "
                f"Confidence: {data['confidence']:.2f} | "
                f"Latency: {data['latency']} ms"
            )

        if not tokens:
            return

        sequence = sequence_signs(tokens)
        self.english_animation_view.disable_live_pose()
        self.english_animation_view.play(sequence)
        duration_ms = int(sum(e.duration for e in sequence) * 1000)
        QTimer.singleShot(duration_ms, self.english_animation_view.enable_live_pose)

    def on_translation_error(self, message):
        print(f"[UI] translation error: {message}")
        self.record_button.setEnabled(True)
        self.english_status_label.setText(f"Error: {message}")

    def on_camera_pose(self, pose: dict):
        self.english_animation_view.set_live_pose(pose)
        self.reverse_animation_view.set_live_pose(pose)

    def on_camera_token(self, token: str, confidence: float):
        self.camera_label.setText(
            f"Camera ASL Input: {token} (conf {confidence:.2f})"
        )

    def on_camera_sequence(self, tokens: list):
        if not tokens:
            return
        self.camera_label.setText(f"Camera ASL Input: {' '.join(tokens)}")
        reverse = asl_to_english(tokens=tokens)
        if reverse.error:
            self.reverse_label.setText("English Output:")
            self.reverse_status_label.setText(
                f"Status: Camera reverse error: {reverse.error}"
            )
            return

        self.reverse_label.setText(f"English Output: {reverse.english_text}")
        self.reverse_status_label.setText(
            f"Status: Camera ASL -> English | "
            f"Confidence: {reverse.confidence:.2f} | "
            f"Latency: {reverse.latency_ms} ms"
        )

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
                self.english_status_label.setText(f"Status: {result.error}")
            return

        self.english_tokens_label.setText(f"ASL Output: {' '.join(tokens)}")
        self.english_status_label.setText(
            f"Status: Live Speech | Confidence: {result.confidence:.2f}"
        )

        sequence = sequence_signs(tokens)
        self.english_animation_view.disable_live_pose()
        self.english_animation_view.play(sequence)
        duration_ms = int(sum(e.duration for e in sequence) * 1000)

        QTimer.singleShot(duration_ms, self.english_animation_view.enable_live_pose)

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
            self.english_status_label.setText("Status: Idle")
        else:
            self.english_status_label.setText("Status: STT warmup failed")

    def set_mode(self, mode: str):
        if mode not in {"english_to_asl", "asl_to_english"}:
            return
        self.mode = mode
        self.mode_stack.setCurrentIndex(0 if mode == "english_to_asl" else 1)
        self._refresh_mode_buttons()

    def _refresh_mode_buttons(self):
        active_style = "font-size: 16px; font-weight: 700; height: 44px;"
        inactive_style = "font-size: 16px; font-weight: 500; height: 44px;"
        if self.mode == "english_to_asl":
            self.english_mode_button.setStyleSheet(active_style)
            self.reverse_mode_button.setStyleSheet(inactive_style)
        else:
            self.english_mode_button.setStyleSheet(inactive_style)
            self.reverse_mode_button.setStyleSheet(active_style)

    def _build_english_to_asl_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout()

        self.english_animation_view = ASLAnimationView()
        self.english_animation_view.setMinimumHeight(620)
        self.english_status_label = QLabel("Status: Idle")
        self.english_status_label.setStyleSheet("font-size: 18px;")
        self.english_tokens_label = QLabel("ASL Output:")
        self.english_tokens_label.setStyleSheet("font-size: 22px;")

        self.record_button = QPushButton("Record")
        self.record_button.setStyleSheet("font-size: 20px; height: 60px;")
        self.record_button.clicked.connect(self.on_record_clicked)
        self.record_button.setEnabled(False)

        page_layout.addWidget(self.english_animation_view, 5)
        page_layout.addWidget(self.english_status_label)
        page_layout.addWidget(self.english_tokens_label)
        page_layout.addWidget(self.record_button)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)
        page.setLayout(page_layout)
        return page

    def _build_asl_to_english_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout()

        self.reverse_animation_view = ASLAnimationView()
        self.reverse_animation_view.setMinimumHeight(620)
        self.reverse_status_label = QLabel("Status: Camera listening...")
        self.reverse_status_label.setStyleSheet("font-size: 18px;")
        self.camera_label = QLabel("Camera ASL Input:")
        self.camera_label.setStyleSheet("font-size: 18px;")
        self.reverse_label = QLabel("English Output:")
        self.reverse_label.setStyleSheet("font-size: 22px;")

        page_layout.addWidget(self.reverse_animation_view, 5)
        page_layout.addWidget(self.reverse_status_label)
        page_layout.addWidget(self.camera_label)
        page_layout.addWidget(self.reverse_label)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(12)
        page.setLayout(page_layout)
        return page
