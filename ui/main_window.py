from core.engine import english_to_asl, asl_to_english, _get_stt_backend
from core.mic_utils import record_audio
from core.audio.vosk_listener import VoskListener
import threading
import subprocess
import sys
import shutil
import os
from datetime import datetime
from PySide6.QtCore import QThread, Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QGuiApplication
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
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
)

try:
    from PySide6.QtTextToSpeech import QTextToSpeech
except Exception:
    QTextToSpeech = None


def _env_int(name: str, default: int, min_value: int | None = None, max_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    if min_value is not None and value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


class MainWindow(QWidget):
    speech_text_received = Signal(str)
    record_result_received = Signal(dict)
    record_error_received = Signal(str)
    stt_warmup_received = Signal(bool)
    camera_sequence_received = Signal(list)
    camera_token_received = Signal(str, float)
    camera_frame_received = Signal(object)
    camera_debug_received = Signal(str, float, int)
    camera_error_received = Signal(str)
    camera_reset_requested = Signal()

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.vosk_init_in_progress = False
        self.camera_thread = None
        self.camera_worker = None
        self.camera_running = False
        self.pending_camera_tokens = []
        self.camera_error_message = None
        self.demo_mode = True
        self.latest_translation_text = ""
        self.tts_engine = self._init_tts_engine()
        self.tts_backend = self._resolve_tts_backend()
        self.espeak_amplitude = _env_int("ASL_TTS_AMPLITUDE", 180, min_value=0, max_value=200)
        self.espeak_speed = _env_int("ASL_TTS_SPEED", 160, min_value=80, max_value=300)
        self.speaker_enabled = self.tts_backend != "none"
        self.compact_ui = self._detect_compact_ui()
        self.preview_min_height = 360 if self.compact_ui else 520
        self.primary_button_height = 48 if self.compact_ui else 60
        self.secondary_button_height = 34 if self.compact_ui else 40
        self.mini_button_height = 30 if self.compact_ui else 36
        self.body_font = 15 if self.compact_ui else 18
        self.heading_font = 18 if self.compact_ui else 22
        self.small_font = 12 if self.compact_ui else 14

        self.setWindowTitle("English <-> ASL Translator")
        self.setMinimumSize(760, 440)
        self.resize(960 if self.compact_ui else 1280, 540 if self.compact_ui else 820)
        self._apply_theme()

        layout = QVBoxLayout()

        self.mode = "english_to_asl"

        mode_row = QHBoxLayout()
        self.mode_badge_label = QLabel("Mode: English -> ASL")
        self.mode_badge_label.setStyleSheet(
            f"font-size: {self.small_font}px; font-weight: 700; color: #d7f9ff; "
            "background-color: #18435a; border-radius: 10px; padding: 6px 10px;"
        )
        self.english_mode_button = QPushButton(
            "E -> ASL" if self.compact_ui else "English -> ASL"
        )
        self.reverse_mode_button = QPushButton(
            "ASL -> EN" if self.compact_ui else "ASL -> English"
        )
        self.demo_mode_button = QPushButton("Demo Mode: ON")
        self.english_mode_button.setObjectName("modeButton")
        self.reverse_mode_button.setObjectName("modeButton")
        self.demo_mode_button.setObjectName("demoButton")
        self.english_mode_button.clicked.connect(
            lambda: self.set_mode("english_to_asl")
        )
        self.reverse_mode_button.clicked.connect(
            lambda: self.set_mode("asl_to_english")
        )
        self.demo_mode_button.clicked.connect(self.toggle_demo_mode)
        self.demo_mode_button.setStyleSheet(
            f"font-size: {self.small_font}px; font-weight: 700; height: {self.mini_button_height}px; "
            "background-color: #204d28; color: #e6ffe9;"
        )
        mode_row.addWidget(self.mode_badge_label)
        mode_row.addWidget(self.english_mode_button)
        mode_row.addWidget(self.reverse_mode_button)
        mode_row.addStretch(1)
        mode_row.addWidget(self.demo_mode_button)
        layout.addLayout(mode_row)

        state_row = QHBoxLayout()
        self.mic_state_label = QLabel("Mic: Warming")
        self.camera_state_label = QLabel("Camera: Starting")
        for label in (self.mic_state_label, self.camera_state_label):
            label.setStyleSheet(
                f"font-size: {self.small_font}px; color: #d0d7de; "
                "background-color: #2a2f36; border-radius: 9px; padding: 5px 9px;"
            )
            state_row.addWidget(label)
        state_row.addStretch(1)
        layout.addLayout(state_row)

        self.mode_stack = QStackedWidget()
        self.mode_stack.addWidget(self._build_english_to_asl_page())
        self.mode_stack.addWidget(self._build_asl_to_english_page())
        layout.addWidget(self.mode_stack, 1)

        if self.compact_ui:
            layout.setContentsMargins(10, 8, 10, 8)
            layout.setSpacing(8)
            self.mode_badge_label.hide()
        else:
            layout.setContentsMargins(24, 16, 24, 16)
            layout.setSpacing(12)
        self.setLayout(layout)
        # English -> ASL avatar should not mirror camera pose.
        self.english_animation_view.disable_live_pose()
        self._refresh_mode_buttons()
        self._apply_demo_mode()

        self.speech_text_received.connect(self.on_speech)
        self.record_result_received.connect(self.on_translation_finished)
        self.record_error_received.connect(self.on_translation_error)
        self.stt_warmup_received.connect(self._on_stt_warmup_complete)
        self.camera_sequence_received.connect(self.on_camera_sequence)
        self.camera_token_received.connect(self.on_camera_token)
        self.camera_frame_received.connect(self.on_camera_frame)
        self.camera_debug_received.connect(self.on_camera_debug)
        self.camera_error_received.connect(self.on_camera_error)
        self.camera_reset_requested.connect(self.on_camera_reset_requested)

        self.start_camera()

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

        self.english_tokens_label.setText(
            f"Detected ASL Tokens: {' '.join(tokens) if tokens else '(none)'}"
        )
        if error:
            self.english_status_label.setText(f"Status: Error ({error})")
        else:
            if self.demo_mode:
                self.english_status_label.setText(
                    f"Heard: {heard_text or '(none)'}"
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

    def on_translation_error(self, message):
        print(f"[UI] translation error: {message}")
        self.record_button.setEnabled(True)
        self.english_status_label.setText(f"Error: {message}")

    def on_camera_pose(self, pose: dict):
        self.english_animation_view.set_live_pose(pose)

    def on_camera_frame(self, frame_image):
        if frame_image is None:
            return
        pixmap = QPixmap.fromImage(frame_image)
        if self.camera_feed_label.width() > 0 and self.camera_feed_label.height() > 0:
            pixmap = pixmap.scaled(
                self.camera_feed_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.camera_feed_label.setPixmap(pixmap)

    def on_camera_token(self, token: str, confidence: float):
        if not self.pending_camera_tokens or self.pending_camera_tokens[-1] != token:
            self.pending_camera_tokens.append(token)
        if self.demo_mode:
            self.reverse_status_label.setText(f"Status: Detecting sign... {token}")
        else:
            self.reverse_status_label.setText(
                f"Status: Detecting sign... {token} (conf {confidence:.2f})"
            )

    def on_camera_debug(self, token: str, confidence: float, streak: int):
        token_text = token if token else "(none)"
        self.reverse_debug_label.setText(
            f"Debug Match: {token_text} | conf={confidence:.2f} | streak={streak}"
        )

    def on_camera_error(self, message: str):
        self.camera_error_message = message
        self.reverse_status_label.setText(f"Status: Camera error: {message}")
        if self.camera_running:
            self.stop_camera()

    def on_camera_sequence(self, tokens: list):
        if not tokens:
            return
        self._finalize_camera_translation(tokens)

    def _finalize_camera_translation(self, tokens: list):
        self.camera_label.setText(
            f"Detected ASL Tokens: {' '.join(tokens)}"
        )
        reverse = asl_to_english(tokens=tokens)
        if reverse.error:
            self.reverse_label.setText("English Translation:")
            self.reverse_status_label.setText(
                f"Status: Camera reverse error: {reverse.error}"
            )
            return

        self.latest_translation_text = reverse.english_text
        self.reverse_label.setText(f"English Translation: {self.latest_translation_text}")
        if self.demo_mode:
            self.reverse_status_label.setText("Status: Translation updated")
        else:
            self.reverse_status_label.setText(
                f"Status: Camera ASL -> English | "
                f"Confidence: {reverse.confidence:.2f} | "
                f"Latency: {reverse.latency_ms} ms"
            )
        self._speak_translation_if_enabled(self.latest_translation_text)
        self.pending_camera_tokens.clear()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        if getattr(self, "vosk", None) is not None:
            self.vosk.stop()

        self.stop_camera()
        event.accept()

    def on_speech(self, text: str):
        print("Heard:", text)

        result = english_to_asl(text=text)
        tokens = result.asl_tokens
        if not tokens:
            if result.error:
                self.english_status_label.setText(f"Status: {result.error}")
            return

        self.english_tokens_label.setText(f"Detected ASL Tokens: {' '.join(tokens)}")
        if self.demo_mode:
            self.english_status_label.setText("Status: Live speech detected")
        else:
            self.english_status_label.setText(
                f"Status: Live Speech | Confidence: {result.confidence:.2f}"
            )

        sequence = sequence_signs(tokens)
        self.english_animation_view.disable_live_pose()
        self.english_animation_view.play(sequence)

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
            self.mic_state_label.setText("Mic: Ready")
        else:
            self.english_status_label.setText("Status: STT warmup failed")
            self.mic_state_label.setText("Mic: Error")

    def set_mode(self, mode: str):
        if mode not in {"english_to_asl", "asl_to_english"}:
            return
        self.mode = mode
        self.mode_stack.setCurrentIndex(0 if mode == "english_to_asl" else 1)
        self._refresh_mode_buttons()
        if mode == "english_to_asl":
            self.mode_badge_label.setText("Mode: English -> ASL")
        else:
            self.mode_badge_label.setText("Mode: ASL -> English")

    def _refresh_mode_buttons(self):
        english_active = self.mode == "english_to_asl"
        self.english_mode_button.setProperty("active", english_active)
        self.reverse_mode_button.setProperty("active", not english_active)
        for button in (self.english_mode_button, self.reverse_mode_button):
            button.style().unpolish(button)
            button.style().polish(button)

    def _build_english_to_asl_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout()

        self.english_animation_view = ASLAnimationView()
        self.english_animation_view.setMinimumHeight(self.preview_min_height)
        self.english_animation_view.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.english_status_label = QLabel("Status: Idle")
        self.english_status_label.setStyleSheet(f"font-size: {self.body_font}px;")
        self.english_tokens_label = QLabel("Detected ASL Tokens:")
        self.english_tokens_label.setStyleSheet(f"font-size: {self.heading_font}px;")

        self.record_button = QPushButton("Record")
        self.record_button.setObjectName("primaryButton")
        self.record_button.setMinimumHeight(self.primary_button_height)
        self.record_button.clicked.connect(self.on_record_clicked)
        self.record_button.setEnabled(False)

        controls_panel = QWidget()
        controls_panel.setObjectName("bottomPanel")
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setSpacing(8 if self.compact_ui else 10)
        controls_layout.addWidget(self.english_status_label)
        controls_layout.addWidget(self.english_tokens_label)
        controls_layout.addWidget(self.record_button)
        controls_panel.setLayout(controls_layout)

        page_layout.addWidget(self.english_animation_view, 1)
        page_layout.addWidget(controls_panel, 0)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8 if self.compact_ui else 12)
        page.setLayout(page_layout)
        return page

    def _build_asl_to_english_page(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout()

        self.camera_feed_label = QLabel("Camera feed")
        self.camera_feed_label.setMinimumHeight(self.preview_min_height)
        self.camera_feed_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.camera_feed_label.setAlignment(Qt.AlignCenter)
        self.camera_feed_label.setStyleSheet(
            "background-color: #111822; border: 1px solid #2d3a4b; "
            f"border-radius: 10px; font-size: {self.body_font}px;"
        )
        self.reverse_status_label = QLabel("Status: Camera listening...")
        self.reverse_status_label.setStyleSheet(f"font-size: {self.body_font}px;")
        self.reverse_debug_label = QLabel("Debug Match: (none) | conf=0.00 | streak=0")
        self.reverse_debug_label.setStyleSheet(f"font-size: {self.small_font}px; color: #b8b8b8;")
        self.camera_label = QLabel("Detected ASL Tokens:")
        self.camera_label.setStyleSheet(f"font-size: {self.body_font + 2}px;")
        self.reverse_label = QLabel("English Translation:")
        self.reverse_label.setWordWrap(True)
        self.reverse_label.setStyleSheet(f"font-size: {self.heading_font}px;")
        self.camera_toggle_button = QPushButton("Stop Camera")
        self.camera_toggle_button.setObjectName("primaryButton")
        self.camera_toggle_button.setMinimumHeight(self.secondary_button_height)
        self.camera_toggle_button.clicked.connect(self.toggle_camera)
        self.reset_translation_button = QPushButton("Reset Translation")
        self.reset_translation_button.setObjectName("secondaryButton")
        self.reset_translation_button.setMinimumHeight(self.secondary_button_height)
        self.reset_translation_button.clicked.connect(self.reset_translation)
        self.add_history_button = QPushButton("Add to History")
        self.add_history_button.setObjectName("secondaryButton")
        self.add_history_button.setMinimumHeight(self.mini_button_height)
        self.add_history_button.clicked.connect(self.add_current_translation_to_history)
        self.speaker_button = QPushButton(
            "Speaker: ON" if self.speaker_enabled else "Speaker: OFF"
        )
        self.speaker_button.setObjectName("secondaryButton")
        self.speaker_button.setMinimumHeight(self.mini_button_height)
        self.speaker_button.clicked.connect(self.toggle_speaker)
        self.test_speaker_button = QPushButton("Test Speaker")
        self.test_speaker_button.setObjectName("secondaryButton")
        self.test_speaker_button.setMinimumHeight(self.mini_button_height)
        self.test_speaker_button.clicked.connect(self.test_speaker)
        if self.tts_backend == "none":
            self.speaker_button.setEnabled(False)
            self.speaker_button.setText("Speaker: Unavailable")
            self.test_speaker_button.setEnabled(False)

        self.history_label = QLabel("History")
        self.history_label.setStyleSheet(f"font-size: {self.body_font}px; font-weight: 700;")
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(52 if self.compact_ui else 70)
        self.history_list.setMaximumHeight(78 if self.compact_ui else 96)
        self.history_list.itemClicked.connect(self.on_history_item_clicked)
        self.history_list.setStyleSheet(
            "background-color: #0a1118; border: 1px solid #1f2a36; border-radius: 8px;"
        )

        controls_panel = QWidget()
        controls_panel.setObjectName("bottomPanel")
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setSpacing(4 if self.compact_ui else 6)
        controls_layout.addWidget(self.reverse_status_label)
        controls_layout.addWidget(self.reverse_debug_label)
        controls_layout.addWidget(self.camera_label)
        controls_layout.addWidget(self.reverse_label)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(6 if self.compact_ui else 10)
        actions_row.addWidget(self.camera_toggle_button, 1)
        actions_row.addWidget(self.reset_translation_button, 1)
        controls_layout.addLayout(actions_row)
        review_row = QHBoxLayout()
        review_row.setSpacing(6 if self.compact_ui else 10)
        review_row.addWidget(self.add_history_button, 1)
        review_row.addWidget(self.speaker_button, 1)
        review_row.addWidget(self.test_speaker_button, 1)
        controls_layout.addLayout(review_row)
        controls_layout.addWidget(self.history_label)
        controls_layout.addWidget(self.history_list)

        controls_panel.setLayout(controls_layout)

        page_layout.addWidget(self.camera_feed_label, 1)
        page_layout.addWidget(controls_panel, 0)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8 if self.compact_ui else 12)
        page.setLayout(page_layout)
        return page

    def start_camera(self):
        if self.camera_running:
            return
        self.camera_error_message = None
        self.pending_camera_tokens.clear()
        if hasattr(self, "camera_label"):
            self.camera_label.setText("Detected ASL Tokens:")
        if hasattr(self, "reverse_label"):
            self.reverse_label.setText("English Translation:")
        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.pose_ready.connect(self.on_camera_pose)
        self.camera_worker.frame_ready.connect(self.camera_frame_received.emit)
        self.camera_worker.token_ready.connect(self.camera_token_received.emit)
        self.camera_worker.debug_ready.connect(self.camera_debug_received.emit)
        self.camera_worker.error_ready.connect(self.camera_error_received.emit)
        self.camera_worker.sequence_ready.connect(self.camera_sequence_received.emit)
        self.camera_reset_requested.connect(self.camera_worker.reset_recognition_state)
        self.camera_thread.start()
        self.camera_running = True
        self.camera_state_label.setText("Camera: Ready")
        if hasattr(self, "camera_toggle_button"):
            self.camera_toggle_button.setText("Stop Camera")
        if hasattr(self, "reverse_status_label"):
            self.reverse_status_label.setText("Status: Camera listening...")
        if hasattr(self, "reverse_debug_label"):
            self.reverse_debug_label.setText("Debug Match: (none) | conf=0.00 | streak=0")
        if hasattr(self, "camera_feed_label"):
            self.camera_feed_label.setText("Camera feed")
            self.camera_feed_label.setPixmap(QPixmap())

    def stop_camera(self):
        if not self.camera_running:
            return
        translated_on_stop = False
        if self.pending_camera_tokens:
            self._finalize_camera_translation(list(self.pending_camera_tokens))
            translated_on_stop = True
        worker = self.camera_worker
        thread = self.camera_thread
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
        if thread is not None:
            try:
                self.camera_reset_requested.disconnect(worker.reset_recognition_state)
            except Exception:
                pass
            thread.requestInterruption()
            thread.quit()
            thread.wait(2500)
        self.camera_thread = None
        self.camera_worker = None
        self.camera_running = False
        self.camera_state_label.setText("Camera: Stopped")
        if hasattr(self, "camera_toggle_button"):
            self.camera_toggle_button.setText("Start Camera")
        if hasattr(self, "reverse_status_label") and not translated_on_stop:
            if self.camera_error_message:
                self.reverse_status_label.setText(
                    f"Status: Camera error: {self.camera_error_message}"
                )
            else:
                self.reverse_status_label.setText("Status: Camera stopped")
        if hasattr(self, "reverse_debug_label"):
            self.reverse_debug_label.setText("Debug Match: (none) | conf=0.00 | streak=0")
        if hasattr(self, "camera_feed_label"):
            self.camera_feed_label.setPixmap(QPixmap())
            self.camera_feed_label.setText("Camera stopped")

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def reset_translation(self):
        self.pending_camera_tokens.clear()
        self.camera_label.setText("Detected ASL Tokens:")
        self.reverse_label.setText("English Translation:")
        self.latest_translation_text = ""
        self.reverse_debug_label.setText("Debug Match: (none) | conf=0.00 | streak=0")
        if self.camera_running:
            self.reverse_status_label.setText("Status: Camera listening...")
            self.camera_reset_requested.emit()
        else:
            self.reverse_status_label.setText("Status: Camera stopped")

    def add_current_translation_to_history(self):
        text = (self.latest_translation_text or "").strip()
        if not text:
            self.reverse_status_label.setText("Status: No translation to add")
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self.history_list.insertItem(0, f"[{stamp}] {text}")
        self.reverse_status_label.setText("Status: Added to history")

    def on_history_item_clicked(self, item: QListWidgetItem):
        if self.tts_backend == "none":
            self.reverse_status_label.setText("Status: Speaker unavailable")
            return
        raw = item.text().strip()
        text = raw
        if raw.startswith("[") and "] " in raw:
            text = raw.split("] ", 1)[1].strip()
        if not text:
            self.reverse_status_label.setText("Status: Empty history item")
            return
        self.latest_translation_text = text
        self.reverse_label.setText(f"English Translation: {text}")
        self._speak_text(text)
        self.reverse_status_label.setText(
            f"Status: Playing history item ({self.tts_backend.upper()})"
        )

    def _init_tts_engine(self):
        if QTextToSpeech is None:
            return None
        try:
            engine = QTextToSpeech(self)
            # Keep output clear and audible for demo use.
            engine.setVolume(1.0)
            engine.setRate(-0.1)
            return engine
        except Exception:
            return None

    def _resolve_tts_backend(self) -> str:
        if sys.platform.startswith("win"):
            return "sapi"
        if shutil.which("espeak") or shutil.which("espeak-ng"):
            return "espeak"
        if self.tts_engine is not None:
            return "qt"
        return "none"

    def toggle_speaker(self):
        if self.tts_backend == "none":
            self.reverse_status_label.setText("Status: Speaker unavailable")
            return
        self.speaker_enabled = not self.speaker_enabled
        if self.speaker_enabled:
            self.speaker_button.setText("Speaker: ON")
            self.reverse_status_label.setText(
                f"Status: Speaker enabled ({self.tts_backend.upper()})"
            )
            self._speak_text("Speaker enabled")
        else:
            self.speaker_button.setText("Speaker: OFF")
            self.reverse_status_label.setText("Status: Speaker disabled")

    def _speak_translation_if_enabled(self, text: str):
        if not self.speaker_enabled:
            return
        if not text:
            return
        self._speak_text(text)

    def _speak_text(self, text: str):
        if not text:
            return
        if self.tts_backend == "qt" and self.tts_engine is not None:
            try:
                self.tts_engine.stop()
                self.tts_engine.say(text)
                return
            except Exception:
                pass
        if self.tts_backend == "sapi":
            safe_text = text.replace("'", "''")
            ps = (
                "Add-Type -AssemblyName System.Speech; "
                "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$s.Volume = 100; $s.Rate = 0; "
                f"$s.Speak('{safe_text}')"
            )
            try:
                subprocess.Popen(
                    ["powershell", "-NoProfile", "-Command", ps],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception:
                pass
        if self.tts_backend == "espeak":
            tts_cmd = shutil.which("espeak-ng") or shutil.which("espeak")
            if tts_cmd:
                try:
                    subprocess.Popen(
                        [
                            tts_cmd,
                            "-a",
                            str(self.espeak_amplitude),
                            "-s",
                            str(self.espeak_speed),
                            text,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except Exception:
                    pass
        self.reverse_status_label.setText("Status: Speaker error")

    def test_speaker(self):
        if self.tts_backend == "none":
            self.reverse_status_label.setText("Status: Speaker unavailable")
            return
        self._speak_text("This is a speaker test.")
        self.reverse_status_label.setText(
            f"Status: Speaker test sent ({self.tts_backend.upper()})"
        )

    def on_camera_reset_requested(self):
        # no-op local slot to keep signal visible and future extensible.
        return

    def toggle_demo_mode(self):
        self.demo_mode = not self.demo_mode
        self._apply_demo_mode()

    def _apply_demo_mode(self):
        if self.demo_mode:
            self.demo_mode_button.setText("Demo Mode: ON")
            self.demo_mode_button.setStyleSheet(
                f"font-size: {self.small_font}px; font-weight: 700; height: {self.mini_button_height}px; "
                "background-color: #204d28; color: #e6ffe9;"
            )
            self.reverse_debug_label.hide()
        else:
            self.demo_mode_button.setText("Demo Mode: OFF")
            self.demo_mode_button.setStyleSheet(
                f"font-size: {self.small_font}px; font-weight: 700; height: {self.mini_button_height}px; "
                "background-color: #4d2b20; color: #ffe9e6;"
            )
            self.reverse_debug_label.show()

    def _detect_compact_ui(self) -> bool:
        touch_env = os.getenv("ASL_TOUCH_UI", "").strip().lower()
        if touch_env in {"1", "true", "yes", "on"}:
            return True
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return False
        geo = screen.availableGeometry()
        return geo.width() <= 1024 or geo.height() <= 600

    def _apply_theme(self):
        mode_font = 14 if self.compact_ui else 16
        mode_height = 36 if self.compact_ui else 44
        demo_height = self.mini_button_height
        css = """
            QWidget {
                background-color: #0f141b;
                color: #e8edf3;
                font-family: "Segoe UI", "Inter", sans-serif;
            }
            QLabel {
                color: #e8edf3;
            }
            QStackedWidget {
                background-color: transparent;
                border: none;
            }
            QPushButton {
                background-color: #202833;
                border: 1px solid #2e3a49;
                border-radius: 10px;
                color: #e8edf3;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background-color: #293445;
                border-color: #3f5065;
            }
            QPushButton:pressed {
                background-color: #1a222d;
            }
            QPushButton#modeButton {
                font-size: __MODE_FONT__px;
                font-weight: 600;
                min-height: __MODE_HEIGHT__px;
            }
            QPushButton#modeButton[active="true"] {
                background-color: #1f6feb;
                border-color: #2f81f7;
                color: #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryButton {
                background-color: #1f6feb;
                border-color: #2f81f7;
                color: #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background-color: #2f81f7;
            }
            QPushButton#primaryButton:pressed {
                background-color: #1a5ec0;
            }
            QPushButton#secondaryButton {
                background-color: #2b3441;
                border-color: #3a4657;
                color: #d7dee7;
                font-weight: 600;
            }
            QPushButton#secondaryButton:hover {
                background-color: #364255;
            }
            QPushButton#demoButton {
                border-radius: 10px;
                min-height: __DEMO_HEIGHT__px;
            }
            QWidget#bottomPanel {
                background-color: #0b121a;
                border: 1px solid #1f2a36;
                border-radius: 12px;
            }
            """
        css = css.replace("__MODE_FONT__", str(mode_font))
        css = css.replace("__MODE_HEIGHT__", str(mode_height))
        css = css.replace("__DEMO_HEIGHT__", str(demo_height))
        self.setStyleSheet(css)
