from core.engine import english_to_asl, asl_to_english, _get_stt_backend
from core.battery_status import read_battery_status
from core.mic_utils import record_audio_until_stop
from core.audio.vosk_listener import VoskListener
import json
import threading
import subprocess
import sys
import shutil
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import QThread, Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve, QEvent
from PySide6.QtGui import QPixmap, QGuiApplication, QPainter, QPen, QColor, QFontMetrics
from core.sequencing.sign_sequencer import sequence_signs, SignEvent
from ui.widgets.animation_view import ASLAnimationView
from ui.worker_camera import CameraWorker
from ui.animation.clip_loader import (
    set_demo_mode as set_demo_clip_mode,
    is_demo_clip_source_active,
    DEMO_CLIP_DIR,
)
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QSizePolicy,
    QListWidget,
    QListWidgetItem,
    QTabWidget,
    QFileDialog,
    QMenu,
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


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_token_set(name: str) -> set[str]:
    raw = os.getenv(name, "")
    if not raw:
        return set()
    tokens: set[str] = set()
    for part in raw.replace(";", ",").split(","):
        token = part.strip().upper()
        if token:
            tokens.add(token)
    return tokens


DEFAULT_DEMO_VOCAB = {
    "HELLO",
    "HOW",
    "ARE",
    "YOU",
    "I",
    "ME",
    "MY",
    "NAME",
    "THANK",
    "PLEASE",
    "GOOD",
    "MORNING",
    "NICE",
    "TO",
    "MEET",
}


class MainWindow(QWidget):
    speech_text_received = Signal(str)
    record_live_text_received = Signal(str)
    record_result_received = Signal(dict)
    record_error_received = Signal(str)
    stt_warmup_received = Signal(bool)
    camera_sequence_received = Signal(list)
    camera_token_received = Signal(str, float)
    camera_frame_received = Signal(object)
    camera_debug_received = Signal(str, float, int)
    camera_error_received = Signal(str)
    camera_reset_requested = Signal()
    camera_stop_requested = Signal()

    def __init__(self):
        super().__init__()
        self.thread = None
        self.worker = None
        self.record_stop_event = None
        self.recording_in_progress = False
        self.record_live_text = ""
        self.stt_ready = False
        self.last_english_sequence = None
        self.last_english_entry = None
        self.english_history_entries = []
        self.english_favorite_entries = []
        self.max_english_history = 40
        self.english_drawer_open = False
        self.english_drawer_width = 420
        self.english_state_path = (
            Path(__file__).resolve().parents[1] / "data" / "english_asl_ui_state.json"
        )
        self.vosk_init_in_progress = False
        self.camera_thread = None
        self.camera_worker = None
        self.camera_running = False
        self.camera_shutdown_in_progress = False
        self.asl_video_path = None
        self.latest_camera_pose = None
        self.latest_camera_debug_token = ""
        self.latest_camera_debug_confidence = 0.0
        self.pending_camera_tokens = []
        self.camera_error_message = None
        self.demo_mode = True
        self.demo_allowed_tokens = _env_token_set("ASL_DEMO_VOCAB") or set(DEFAULT_DEMO_VOCAB)
        self.latest_translation_text = ""
        self.tts_engine = self._init_tts_engine()
        self.tts_backend = self._resolve_tts_backend()
        self.espeak_amplitude = _env_int("ASL_TTS_AMPLITUDE", 180, min_value=0, max_value=200)
        self.espeak_speed = _env_int("ASL_TTS_SPEED", 160, min_value=80, max_value=300)
        self.tts_test_volume = _env_int("ASL_TTS_TEST_VOLUME", 95, min_value=0, max_value=100)
        self.tts_output_volume = _env_int(
            "ASL_TTS_OUTPUT_VOLUME",
            self.tts_test_volume,
            min_value=0,
            max_value=100,
        )
        self.tts_mixer_card = os.getenv("ASL_TTS_MIXER_CARD", "wm8960soundcard")
        self._last_volume_sync = 0.0
        self.tts_alsa_device = os.getenv(
            "ASL_TTS_ALSA_DEVICE",
            "default:CARD=wm8960soundcard",
        )
        self.fast_camera_mode = _env_bool("ASL_CAMERA_FAST_MODE", False)
        self.speaker_enabled = self.tts_backend != "none"
        self.compact_ui = self._detect_compact_ui()
        self.portrait_ui = self._detect_portrait_ui()
        if self.portrait_ui:
            self.english_drawer_width = 240 if self.compact_ui else 280
            self.preview_min_height = 420 if self.compact_ui else 560
            self.asl_preview_min_height = 640 if self.compact_ui else 860
            self.english_preview_min_height = 720 if self.compact_ui else 980
            self.english_controls_collapsed_height = 150 if self.compact_ui else 170
            self.english_saved_drawer_max_height = 86 if self.compact_ui else 104
            self.reverse_controls_max_height = 130 if self.compact_ui else 150
            self.reverse_history_max_height = 58 if self.compact_ui else 72
        else:
            self.english_drawer_width = 300 if self.compact_ui else 420
            self.preview_min_height = 360 if self.compact_ui else 520
            self.asl_preview_min_height = 560 if self.compact_ui else 780
            self.english_preview_min_height = 620 if self.compact_ui else 900
            self.english_controls_collapsed_height = 140 if self.compact_ui else 160
            self.english_saved_drawer_max_height = 96 if self.compact_ui else 128
            self.reverse_controls_max_height = 120 if self.compact_ui else 140
            self.reverse_history_max_height = 60 if self.compact_ui else 76
        self.english_controls_expanded_height = (
            self.english_controls_collapsed_height + self.english_saved_drawer_max_height + 18
        )
        self.primary_button_height = 48 if self.compact_ui else 60
        self.secondary_button_height = 34 if self.compact_ui else 40
        self.mini_button_height = 30 if self.compact_ui else 36
        self.body_font = 15 if self.compact_ui else 18
        self.heading_font = 18 if self.compact_ui else 22
        self.small_font = 12 if self.compact_ui else 14
        self.battery_timer = QTimer(self)
        self.battery_timer.setInterval(
            _env_int("ASL_BATTERY_POLL_MS", 30000, min_value=1000)
        )
        self.battery_timer.timeout.connect(self.refresh_battery_status)
        self.soft_keyboard_process = None
        self.touch_ui_enabled = os.getenv("ASL_TOUCH_UI", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        self.setWindowTitle("English <-> ASL Translator")
        if self.portrait_ui:
            self.setMinimumSize(420, 760)
            self.resize(720 if self.compact_ui else 820, 1180 if self.compact_ui else 1280)
        else:
            self.setMinimumSize(760, 440)
            self.resize(960 if self.compact_ui else 1280, 540 if self.compact_ui else 820)
        self._apply_theme()

        layout = QVBoxLayout()

        self.mode = "english_to_asl"

        header_panel = QWidget()
        header_panel.setObjectName("topPanel")
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(16, 14, 16, 14)
        header_layout.setSpacing(10 if self.compact_ui else 12)

        title_row = QHBoxLayout()
        title_row.setSpacing(10)
        title_copy = QVBoxLayout()
        title_copy.setSpacing(2)
        self.header_title_label = QLabel("English <-> ASL")
        self.header_title_label.setObjectName("pageTitle")
        self.header_subtitle_label = QLabel("Real-time translation in a clean two-mode workspace")
        self.header_subtitle_label.setObjectName("pageSubtitle")
        title_copy.addWidget(self.header_title_label)
        title_copy.addWidget(self.header_subtitle_label)
        title_row.addLayout(title_copy, 1)

        self.mode_badge_label = QLabel("English to ASL active")
        self.mode_badge_label.setObjectName("modeBadge")
        self.english_mode_button = QPushButton(
            "E -> ASL" if self.compact_ui else "English -> ASL"
        )
        self.reverse_mode_button = QPushButton(
            "ASL -> EN" if self.compact_ui else "ASL -> English"
        )
        self.demo_mode_button = QPushButton("Demo Mode: ON")
        self.close_window_button = QPushButton("X")
        self.english_mode_button.setObjectName("modeButton")
        self.reverse_mode_button.setObjectName("modeButton")
        self.demo_mode_button.setObjectName("demoButton")
        self.close_window_button.setObjectName("closeButton")
        self.english_mode_button.clicked.connect(
            lambda: self.set_mode("english_to_asl")
        )
        self.reverse_mode_button.clicked.connect(
            lambda: self.set_mode("asl_to_english")
        )
        self.demo_mode_button.clicked.connect(self.toggle_demo_mode)
        self.close_window_button.clicked.connect(self.close)
        self.close_window_button.setToolTip("Exit")
        close_button_size = 34 if self.compact_ui else 40
        self.close_window_button.setFixedSize(close_button_size, close_button_size)
        title_row.addWidget(self.mode_badge_label, 0, Qt.AlignTop)
        title_row.addWidget(self.demo_mode_button, 0, Qt.AlignTop)
        title_row.addWidget(self.close_window_button, 0, Qt.AlignTop)
        header_layout.addLayout(title_row)

        mode_shell = QWidget()
        mode_shell.setObjectName("modeShell")
        mode_row = QHBoxLayout()
        mode_row.setContentsMargins(6, 6, 6, 6)
        mode_row.setSpacing(6)
        mode_row.addWidget(self.english_mode_button)
        mode_row.addWidget(self.reverse_mode_button)
        mode_shell.setLayout(mode_row)
        header_layout.addWidget(mode_shell)

        state_row = QHBoxLayout()
        state_row.setSpacing(8)
        self.mic_state_label = QLabel("Mic: Warming")
        self.camera_state_label = QLabel("Camera: Starting")
        self.battery_state_label = QLabel("Battery: Detecting")
        for label in (
            self.mic_state_label,
            self.camera_state_label,
            self.battery_state_label,
        ):
            label.setObjectName("statusChip")
            state_row.addWidget(label)
        state_row.addStretch(1)
        header_layout.addLayout(state_row)
        header_panel.setLayout(header_layout)
        layout.addWidget(header_panel)

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
        self.refresh_battery_status()
        self.battery_timer.start()
        self._sync_linux_playback_volume(force=True)

        self.speech_text_received.connect(self.on_speech)
        self.record_live_text_received.connect(self.on_record_live_text)
        self.record_result_received.connect(self.on_translation_finished)
        self.record_error_received.connect(self.on_translation_error)
        self.stt_warmup_received.connect(self._on_stt_warmup_complete)
        self.camera_sequence_received.connect(self.on_camera_sequence)
        self.camera_token_received.connect(self.on_camera_token)
        self.camera_frame_received.connect(self.on_camera_frame)
        self.camera_debug_received.connect(self.on_camera_debug)
        self.camera_error_received.connect(self.on_camera_error)
        self.camera_reset_requested.connect(self.on_camera_reset_requested)

        self._load_english_collections()
        self._refresh_english_collections_ui()
        self.set_mode(self.mode)

        self.vosk = None
        self.english_status_label.setText("Status: Loading speech model...")
        threading.Thread(target=self._warmup_stt_backend, daemon=True).start()

    def on_record_clicked(self):
        if self.recording_in_progress:
            self.on_stop_record_clicked()
            return

        self.english_status_label.setText("Status: Recording... tap again to stop")
        self.english_tokens_label.setText("ASL Output:")
        self.record_button.setEnabled(True)
        self.replay_button.setEnabled(False)
        self.recording_in_progress = True
        self.record_live_text = ""
        self.record_stop_event = None
        self._refresh_record_toggle_button()

        try:
            listener = self._ensure_vosk_listener()
            listener.on_text = self.record_live_text_received.emit
            listener.latest_text = ""
            listener.start()
        except Exception as e:
            self.recording_in_progress = False
            self.record_button.setEnabled(True)
            self._refresh_record_toggle_button()
            self.record_error_received.emit(str(e))

    def on_stop_record_clicked(self):
        if not self.recording_in_progress:
            return
        self.english_status_label.setText("Status: Finalizing speech...")
        self.recording_in_progress = False
        self.record_button.setEnabled(True)
        self._refresh_record_toggle_button()
        threading.Thread(target=self._run_record_job, daemon=True).start()

    def on_replay_clicked(self):
        if not self.last_english_sequence:
            return
        self.english_status_label.setText("Status: Replaying last animation")
        self._play_english_sequence(self.last_english_sequence)

    def on_add_favorite_clicked(self):
        if not self.last_english_entry:
            self.english_status_label.setText("Status: No animation to favorite yet")
            self._refresh_favorite_button()
            return
        if self._contains_english_entry(self.english_favorite_entries, self.last_english_entry):
            self.english_favorite_entries = [
                entry
                for entry in self.english_favorite_entries
                if self._english_entry_signature(entry) != self._english_entry_signature(self.last_english_entry)
            ]
            self.english_status_label.setText("Status: Removed from favorites")
        else:
            self.english_favorite_entries.insert(0, self.last_english_entry)
            self.english_status_label.setText("Status: Added to favorites")
        self._refresh_english_collections_ui()
        self._save_english_collections()
        self._refresh_favorite_button()

    def on_remove_favorite_clicked(self):
        item = self.english_favorites_list.currentItem()
        if item is None:
            self.english_status_label.setText("Status: Select a favorite to remove")
            return
        row = self.english_favorites_list.row(item)
        if row < 0 or row >= len(self.english_favorite_entries):
            self.english_status_label.setText("Status: Favorite selection invalid")
            return
        del self.english_favorite_entries[row]
        self._refresh_english_collections_ui()
        self._save_english_collections()
        self._refresh_favorite_button()
        self.english_status_label.setText("Status: Favorite removed")

    def on_clear_english_history_clicked(self):
        self.english_history_entries.clear()
        self._refresh_english_collections_ui()
        self._save_english_collections()
        self.english_status_label.setText("Status: History cleared")

    def on_clear_reverse_history_clicked(self):
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        self.reverse_status_label.setText("Status: ASL history cleared")
        self._refresh_asl_settings_actions()

    def on_replay_latest_reverse_history_clicked(self):
        if not hasattr(self, "history_list") or self.history_list.count() == 0:
            self.reverse_status_label.setText("Status: ASL history is empty")
            return
        item = self.history_list.item(0)
        if item is None:
            self.reverse_status_label.setText("Status: ASL history is empty")
            return
        self.on_history_item_clicked(item)

    def on_english_history_item_clicked(self, item: QListWidgetItem):
        row = self.english_history_list.row(item)
        if row < 0 or row >= len(self.english_history_entries):
            self.english_status_label.setText("Status: History item unavailable")
            return
        entry = self.english_history_entries[row]
        self._replay_english_entry(entry, source="history")

    def on_english_favorite_item_clicked(self, item: QListWidgetItem):
        row = self.english_favorites_list.row(item)
        if row < 0 or row >= len(self.english_favorite_entries):
            self.english_status_label.setText("Status: Favorite item unavailable")
            return
        entry = self.english_favorite_entries[row]
        self._replay_english_entry(entry, source="favorites")

    def on_translation_finished(self, data):
        self.recording_in_progress = False
        self.record_stop_event = None
        self.record_button.setEnabled(True)
        self._refresh_record_toggle_button()
        self.replay_button.setEnabled(bool(self.last_english_sequence))
        tokens = data["tokens"]
        heard_text = data.get("text", "")
        error = data.get("error")
        used_fingerspelling = bool(data.get("used_fingerspelling", False))
        spelled_words = [str(w) for w in (data.get("spelled_words") or []) if str(w).strip()]

        self.english_tokens_label.setText(
            f"Detected ASL Tokens: {' '.join(tokens) if tokens else '(none)'}"
        )
        if error:
            self.english_status_label.setText(f"Status: Error ({error})")
        elif used_fingerspelling and spelled_words:
            words = ", ".join(spelled_words[:3])
            extra = f" (+{len(spelled_words) - 3} more)" if len(spelled_words) > 3 else ""
            self.english_status_label.setText(
                f"Status: No sign found for {words}{extra}; spelling letter-by-letter"
            )
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
        self._remember_english_result(
            text=heard_text,
            tokens=tokens,
            sequence=sequence,
        )
        self._play_english_sequence(sequence)

    def on_translation_error(self, message):
        print(f"[UI] translation error: {message}")
        self.recording_in_progress = False
        self.record_stop_event = None
        self.record_button.setEnabled(True)
        self._refresh_record_toggle_button()
        self.replay_button.setEnabled(bool(self.last_english_sequence))
        self.english_status_label.setText(f"Error: {message}")

    def on_record_live_text(self, text: str):
        heard = (text or "").strip()
        if not heard:
            return
        self.record_live_text = heard
        if self.recording_in_progress:
            self.english_status_label.setText(f"Status: Recording... Heard: {heard}")

    def on_camera_pose(self, pose: dict):
        self.latest_camera_pose = pose
        self.english_animation_view.set_live_pose(pose)

    def on_camera_frame(self, frame_image):
        if frame_image is None:
            return
        pixmap = QPixmap.fromImage(frame_image)
        self._draw_camera_overlay(pixmap)
        if self.camera_feed_label.width() > 0 and self.camera_feed_label.height() > 0:
            pixmap = pixmap.scaled(
                self.camera_feed_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        self.camera_feed_label.setPixmap(pixmap)

    def on_camera_token(self, token: str, confidence: float):
        if self.demo_mode and self.demo_allowed_tokens and token not in self.demo_allowed_tokens:
            self.reverse_status_label.setText(f"Status: Demo mode ignored token: {token}")
            return
        if not self.pending_camera_tokens or self.pending_camera_tokens[-1] != token:
            self.pending_camera_tokens.append(token)
        if self.demo_mode:
            self.reverse_status_label.setText("Status: Capturing signs...")
        else:
            self.reverse_status_label.setText(
                f"Status: Capturing signs... ({len(self.pending_camera_tokens)} buffered)"
            )

    def on_camera_debug(self, token: str, confidence: float, streak: int):
        self.latest_camera_debug_token = token or ""
        self.latest_camera_debug_confidence = float(confidence)
        token_text = token if token else "(none)"
        self.reverse_debug_label.setText(
            f"Debug Match: {token_text} | conf={confidence:.2f} | streak={streak}"
        )

    def _draw_camera_overlay(self, pixmap: QPixmap):
        pose = self.latest_camera_pose
        if pose is None or pixmap.isNull():
            return

        hand_boxes = []
        for side in ("left", "right"):
            box = self._hand_overlay_box(pose, side, pixmap.width(), pixmap.height())
            if box is not None:
                hand_boxes.append(box)
        if not hand_boxes:
            return

        max_prominence = max(item[4] for item in hand_boxes)
        min_prominence = max(4.0, max_prominence * 0.65)
        draw_boxes = [item[:4] for item in hand_boxes if item[4] >= min_prominence]
        if not draw_boxes:
            return

        token = self.latest_camera_debug_token.strip()
        confidence = self.latest_camera_debug_confidence
        label = ""
        if token:
            label = f"{token} {confidence * 100:.0f}%"

        painter = QPainter(pixmap)
        try:
            pen = QPen(QColor(57, 210, 122), 3)
            painter.setPen(pen)
            painter.setRenderHint(QPainter.Antialiasing)
            font_metrics = QFontMetrics(painter.font())
            for x, y, w, h in draw_boxes:
                painter.drawRect(x, y, w, h)
                if label:
                    text_w = font_metrics.horizontalAdvance(label) + 12
                    text_h = font_metrics.height() + 6
                    text_x = x
                    text_y = max(0, y - text_h - 4)
                    painter.fillRect(text_x, text_y, text_w, text_h, QColor(15, 23, 42, 210))
                    painter.setPen(QColor(255, 255, 255))
                    painter.drawText(
                        text_x + 6,
                        text_y + text_h - font_metrics.descent() - 3,
                        label,
                    )
                    painter.setPen(pen)
        finally:
            painter.end()

    def _hand_overlay_box(self, pose: dict, side: str, width: int, height: int):
        keys = [
            f"hand_{side}",
            f"{side}_thumb_tip",
            f"{side}_index_tip",
            f"{side}_middle_tip",
            f"{side}_ring_tip",
            f"{side}_pinky_tip",
        ]
        points = []
        for key in keys:
            pt = pose.get(key)
            if pt is None:
                continue
            try:
                x = int(float(pt[0]) * width)
                y = int(float(pt[1]) * height)
            except Exception:
                continue
            points.append((x, y))
        if len(points) < 3:
            return None

        xs = [pt[0] for pt in points]
        ys = [pt[1] for pt in points]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        prominence = float(len(points)) + 0.04 * float(span_x + span_y)
        if len(points) == 1:
            pad = max(24, min(width, height) // 18)
            min_x = xs[0] - pad
            max_x = xs[0] + pad
            min_y = ys[0] - pad
            max_y = ys[0] + pad
        else:
            pad = 18
            min_x = min(xs) - pad
            max_x = max(xs) + pad
            min_y = min(ys) - pad
            max_y = max(ys) + pad

        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(width - 1, max_x)
        max_y = min(height - 1, max_y)
        box_w = max(12, max_x - min_x)
        box_h = max(12, max_y - min_y)
        return min_x, min_y, box_w, box_h, prominence

    def on_camera_error(self, message: str):
        self.camera_error_message = message
        self.reverse_status_label.setText(f"Status: Camera error: {message}")
        if self.camera_running:
            self.stop_camera()

    def on_camera_sequence(self, tokens: list):
        if not tokens:
            return
        if self.demo_mode and self.demo_allowed_tokens:
            filtered = [t for t in tokens if t in self.demo_allowed_tokens]
            if not filtered:
                self.reverse_status_label.setText("Status: Demo mode ignored out-of-vocabulary signs")
                return
            tokens = filtered
        self._finalize_camera_translation(tokens)

    def _finalize_camera_translation(self, tokens: list):
        if self.demo_mode and self.demo_allowed_tokens:
            tokens = [t for t in tokens if t in self.demo_allowed_tokens]
            if not tokens:
                self.reverse_status_label.setText("Status: Demo mode ignored out-of-vocabulary signs")
                return
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
        self.battery_timer.stop()

        if self.thread is not None and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()

        if self.record_stop_event is not None:
            self.record_stop_event.set()

        if getattr(self, "vosk", None) is not None:
            self.vosk.stop()

        self.stop_camera()
        self._hide_soft_keyboard()
        event.accept()

    def eventFilter(self, watched, event):
        if watched is getattr(self, "english_text_input", None):
            if event.type() == QEvent.FocusIn:
                self._show_soft_keyboard()
            elif event.type() == QEvent.FocusOut:
                self._hide_soft_keyboard()
        return super().eventFilter(watched, event)

    def on_speech(self, text: str):
        print("Heard:", text)
        self._process_english_text(text, source="speech")

    def on_typed_text_submit(self):
        if not hasattr(self, "english_text_input"):
            return
        raw = self.english_text_input.text()
        text = (raw or "").strip()
        if not text:
            self.english_status_label.setText("Status: Type text to translate")
            return
        self._process_english_text(text, source="typed")
        self.english_text_input.selectAll()
        self._hide_soft_keyboard()

    def _resolve_soft_keyboard_command(self) -> list[str] | None:
        env_cmd = os.getenv("ASL_SOFT_KEYBOARD_CMD", "").strip()
        if env_cmd:
            return ["/bin/bash", "-lc", env_cmd]
        for command in (
            "squeekboard",
            "wvkbd-mobintl",
            "wvkbd",
            "matchbox-keyboard",
            "onboard",
            "florence",
        ):
            if shutil.which(command):
                return [command]
        return None

    def _set_native_osk_visible(self, visible: bool) -> bool:
        if not shutil.which("gdbus"):
            return False
        try:
            subprocess.run(
                [
                    "gdbus",
                    "call",
                    "--session",
                    "--dest",
                    "sm.puri.OSK0",
                    "--object-path",
                    "/sm/puri/OSK0",
                    "--method",
                    "sm.puri.OSK0.SetVisible",
                    "true" if visible else "false",
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except Exception:
            return False

    def _show_soft_keyboard(self):
        if not self.touch_ui_enabled:
            return
        if self._set_native_osk_visible(True):
            return
        if self.soft_keyboard_process is not None and self.soft_keyboard_process.poll() is None:
            return
        command = self._resolve_soft_keyboard_command()
        if command is None:
            return
        try:
            self.soft_keyboard_process = subprocess.Popen(command)
        except Exception:
            self.soft_keyboard_process = None

    def _hide_soft_keyboard(self):
        self._set_native_osk_visible(False)
        process = self.soft_keyboard_process
        if process is None:
            return
        if process.poll() is not None:
            self.soft_keyboard_process = None
            return
        try:
            process.terminate()
        except Exception:
            pass
        self.soft_keyboard_process = None

    def _process_english_text(self, text: str, source: str):
        text = (text or "").strip()
        if not text:
            return

        result = english_to_asl(text=text)
        tokens = result.asl_tokens
        used_fingerspelling = bool(getattr(result, "used_fingerspelling", False))
        spelled_words = [
            str(w) for w in (getattr(result, "spelled_words", None) or []) if str(w).strip()
        ]
        if not tokens:
            if result.error:
                self.english_status_label.setText(f"Status: {result.error}")
            else:
                self.english_status_label.setText("Status: No ASL tokens detected")
            return

        self.english_tokens_label.setText(f"Detected ASL Tokens: {' '.join(tokens)}")
        if used_fingerspelling and spelled_words:
            words = ", ".join(spelled_words[:3])
            extra = f" (+{len(spelled_words) - 3} more)" if len(spelled_words) > 3 else ""
            self.english_status_label.setText(
                f"Status: No sign found for {words}{extra}; spelling letter-by-letter"
            )
        elif source == "typed":
            if self.demo_mode:
                self.english_status_label.setText(f"Status: Typed text translated | {text}")
            else:
                self.english_status_label.setText(
                    f"Status: Typed Text | Confidence: {result.confidence:.2f}"
                )
        elif self.demo_mode:
            self.english_status_label.setText("Status: Live speech detected")
        else:
            self.english_status_label.setText(
                f"Status: Live Speech | Confidence: {result.confidence:.2f}"
            )

        sequence = sequence_signs(tokens)
        self._remember_english_result(
            text=text,
            tokens=tokens,
            sequence=sequence,
        )
        self._play_english_sequence(sequence)

    def _play_english_sequence(self, sequence):
        if not sequence:
            return
        self.last_english_sequence = list(sequence)
        self.replay_button.setEnabled(True)
        self._refresh_favorite_button()
        self.english_animation_view.disable_live_pose()
        self.english_animation_view.play(sequence)

    def _refresh_record_toggle_button(self):
        is_recording = self.recording_in_progress
        self.record_button.setProperty("recording", is_recording)
        self.record_button.setText("■" if is_recording else "●")
        self.record_button.setToolTip(
            "Stop recording" if is_recording else "Start recording"
        )
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)

    def _remember_english_result(self, text: str, tokens: list, sequence: list):
        if not tokens or not sequence:
            return
        entry = self._make_english_entry(text=text, tokens=tokens, sequence=sequence)
        self.last_english_entry = entry
        self._refresh_favorite_button()
        if self._contains_english_entry(self.english_history_entries[:1], entry):
            return
        self.english_history_entries.insert(0, entry)
        if len(self.english_history_entries) > self.max_english_history:
            self.english_history_entries = self.english_history_entries[: self.max_english_history]
        self._refresh_english_collections_ui()
        self._save_english_collections()

    def _replay_english_entry(self, entry: dict, source: str):
        sequence = self._sequence_from_payload(entry.get("sequence"))
        if not sequence:
            tokens = entry.get("tokens", [])
            sequence = sequence_signs(tokens)
        if not sequence:
            self.english_status_label.setText("Status: Unable to replay item")
            return
        self.last_english_entry = entry
        self._refresh_favorite_button()
        self.english_tokens_label.setText(
            f"Detected ASL Tokens: {' '.join(entry.get('tokens', [])) or '(none)'}"
        )
        text = (entry.get("text") or "").strip()
        if text:
            self.english_status_label.setText(
                f"Status: Replaying {source} item | Heard: {text}"
            )
        else:
            self.english_status_label.setText(f"Status: Replaying {source} item")
        self._play_english_sequence(sequence)

    def _make_english_entry(self, text: str, tokens: list, sequence: list) -> dict:
        now = datetime.now()
        return {
            "stamp": now.strftime("%H:%M:%S"),
            "created_at": now.isoformat(timespec="seconds"),
            "text": (text or "").strip(),
            "tokens": list(tokens),
            "sequence": self._sequence_to_payload(sequence),
        }

    def _sequence_to_payload(self, sequence: list) -> list:
        payload = []
        for evt in sequence:
            payload.append(
                {
                    "token": evt.token,
                    "clip": evt.clip,
                    "start": float(evt.start),
                    "duration": float(evt.duration),
                }
            )
        return payload

    def _sequence_from_payload(self, payload: list) -> list:
        if not isinstance(payload, list):
            return []
        seq = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                seq.append(
                    SignEvent(
                        token=str(item.get("token", "")),
                        clip=str(item.get("clip", "")),
                        start=float(item.get("start", 0.0)),
                        duration=float(item.get("duration", 0.0)),
                    )
                )
            except Exception:
                continue
        return seq

    def _contains_english_entry(self, entries: list, candidate: dict) -> bool:
        for entry in entries:
            if self._english_entry_signature(entry) == self._english_entry_signature(candidate):
                return True
        return False

    def _english_entry_signature(self, entry: dict):
        text = (entry.get("text") or "").strip().lower()
        tokens = tuple(entry.get("tokens", []))
        return text, tokens

    def _format_english_entry(self, entry: dict) -> str:
        stamp = entry.get("stamp") or "??:??:??"
        text = (entry.get("text") or "").strip()
        if text:
            return f"[{stamp}] {text}"
        return f"[{stamp}] {' '.join(entry.get('tokens', [])) or '(no tokens)'}"

    def _refresh_english_collections_ui(self):
        if hasattr(self, "english_history_list"):
            self.english_history_list.clear()
            for entry in self.english_history_entries:
                self.english_history_list.addItem(self._format_english_entry(entry))
        if hasattr(self, "english_favorites_list"):
            self.english_favorites_list.clear()
            for entry in self.english_favorite_entries:
                self.english_favorites_list.addItem(self._format_english_entry(entry))
        self._refresh_favorite_button()

    def _refresh_favorite_button(self):
        if not hasattr(self, "favorite_button"):
            return
        has_current = self.last_english_entry is not None
        is_favorited = (
            has_current
            and self._contains_english_entry(
                self.english_favorite_entries,
                self.last_english_entry,
            )
        )
        self.favorite_button.setEnabled(has_current)
        self.favorite_button.setProperty("favorited", bool(is_favorited))
        self.favorite_button.setText("★" if is_favorited else "☆")
        self.favorite_button.setToolTip(
            "Remove from favorites" if is_favorited else "Add to favorites"
        )
        self.favorite_button.style().unpolish(self.favorite_button)
        self.favorite_button.style().polish(self.favorite_button)

    def toggle_english_saved_drawer(self):
        if not hasattr(self, "english_saved_drawer"):
            return
        opening = not self.english_drawer_open
        if opening:
            self.english_saved_drawer.setVisible(True)
        self._set_english_controls_height(drawer_open=opening)
        target_width = self.english_drawer_width if not self.english_drawer_open else 0
        current_width = self.english_saved_drawer.maximumWidth()
        self.english_saved_drawer_anim.stop()
        self.english_saved_drawer_anim.setStartValue(current_width)
        self.english_saved_drawer_anim.setEndValue(target_width)
        self.english_saved_drawer_anim.start()
        self.english_drawer_open = not self.english_drawer_open
        if self.english_drawer_open:
            self.english_saved_drawer_toggle.setText("Saved ▸")
            self.english_saved_drawer_toggle.setToolTip("Hide saved drawer")
        else:
            self.english_saved_drawer_toggle.setText("Saved ◂")
            self.english_saved_drawer_toggle.setToolTip("Show saved drawer")

    def _on_english_drawer_anim_value(self, value):
        width = int(value)
        self.english_saved_drawer.setMinimumWidth(width)

    def _on_english_drawer_anim_finished(self):
        if not self.english_drawer_open:
            self.english_saved_drawer.setVisible(False)
            self._set_english_controls_height(drawer_open=False)

    def _set_english_controls_height(self, drawer_open: bool):
        if not hasattr(self, "english_controls_panel"):
            return
        target = (
            self.english_controls_expanded_height
            if drawer_open
            else self.english_controls_collapsed_height
        )
        self.english_controls_panel.setMaximumHeight(target)

    def _load_english_collections(self):
        path = self.english_state_path
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            history = data.get("history", [])
            favorites = data.get("favorites", [])
            self.english_history_entries = self._sanitize_english_entries(history)
            self.english_favorite_entries = self._sanitize_english_entries(favorites)
        except Exception as e:
            print(f"[UI] failed loading English collections: {e}")
            self.english_history_entries = []
            self.english_favorite_entries = []

    def _save_english_collections(self):
        path = self.english_state_path
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "history": self.english_history_entries[: self.max_english_history],
                "favorites": self.english_favorite_entries,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[UI] failed saving English collections: {e}")

    def _sanitize_english_entries(self, entries: list) -> list:
        if not isinstance(entries, list):
            return []
        clean = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            text = (entry.get("text") or "").strip()
            tokens = entry.get("tokens", [])
            if not isinstance(tokens, list):
                tokens = []
            tokens = [str(tok) for tok in tokens if str(tok).strip()]
            seq = self._sequence_from_payload(entry.get("sequence", []))
            if not tokens and not seq:
                continue
            clean.append(
                {
                    "stamp": entry.get("stamp") or "??:??:??",
                    "created_at": entry.get("created_at") or "",
                    "text": text,
                    "tokens": tokens,
                    "sequence": self._sequence_to_payload(seq),
                }
            )
        return clean[: self.max_english_history]

    def _start_vosk_async(self):
        if self.vosk is not None:
            return
        if self.vosk_init_in_progress:
            return
        self.vosk_init_in_progress = True
        threading.Thread(target=self._init_vosk_listener, daemon=True).start()

    def _ensure_vosk_listener(self):
        if self.vosk is not None:
            return self.vosk
        backend = _get_stt_backend()
        self.vosk = VoskListener(
            model=backend.model,
            on_text=self.record_live_text_received.emit,
        )
        return self.vosk

    def _run_record_job(self):
        try:
            listener = self._ensure_vosk_listener()
            text = listener.stop(wait=True).strip()
            if not text:
                text = self.record_live_text.strip()
            if not text:
                raise RuntimeError("No speech detected. Try recording again.")
            result = english_to_asl(text=text)
            print(
                "[Record] stt text=",
                repr(text),
                "tokens=",
                result.asl_tokens,
                "error=",
                result.error,
            )
            self.record_result_received.emit({
                "tokens": result.asl_tokens,
                "confidence": result.confidence,
                "latency": result.latency_ms,
                "text": text,
                "error": result.error,
                "used_fingerspelling": bool(getattr(result, "used_fingerspelling", False)),
                "spelled_words": list(getattr(result, "spelled_words", None) or []),
            })
        except Exception as e:
            self.record_error_received.emit(str(e))

    def _init_vosk_listener(self):
        try:
            self._ensure_vosk_listener()
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
        self.stt_ready = ok
        # Keep mic control usable even while warmup is slow/fails; fallback
        # initialization can still occur on-demand when recording starts.
        self.record_button.setEnabled(True)
        self._refresh_record_toggle_button()
        if ok:
            self.english_status_label.setText("Status: Idle")
            self.mic_state_label.setText("Mic: Ready")
            self._start_vosk_async()
        else:
            self.english_status_label.setText("Status: Mic fallback mode (warmup failed)")
            self.mic_state_label.setText("Mic: Retry on record")

    def set_mode(self, mode: str):
        if mode not in {"english_to_asl", "asl_to_english"}:
            return
        self.mode = mode
        self.mode_stack.setCurrentIndex(0 if mode == "english_to_asl" else 1)
        self._refresh_mode_buttons()
        if mode == "english_to_asl":
            self.mode_badge_label.setText("English to ASL active")
            if self.camera_running:
                self.stop_camera(finalize_pending=False)
            else:
                self.camera_state_label.setText(
                    "Video: Standby" if self.asl_video_path else "Camera: Standby"
                )
                if hasattr(self, "camera_toggle_button"):
                    self.camera_toggle_button.setText("Start Input")
                if hasattr(self, "reverse_status_label"):
                    self.reverse_status_label.setText(
                        "Status: Input idle until ASL mode"
                    )
                if hasattr(self, "reverse_debug_label"):
                    self.reverse_debug_label.setText(
                        "Debug Match: (none) | conf=0.00 | streak=0"
                    )
                if hasattr(self, "camera_feed_label"):
                    self.camera_feed_label.setPixmap(QPixmap())
                    self.camera_feed_label.setText(
                        "Video standby" if self.asl_video_path else "Camera standby"
                    )
        else:
            self.mode_badge_label.setText("ASL to English active")
            if not self.camera_running:
                self.start_camera()
        self._update_asl_source_buttons()

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
        self.english_animation_view.setObjectName("previewSurface")
        self.english_animation_view.setMinimumHeight(self.english_preview_min_height)
        self.english_animation_view.setMaximumHeight(self.english_preview_min_height)
        self.english_animation_view.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.english_status_label = QLabel("Status: Idle")
        self.english_status_label.setObjectName("statusLine")
        self.english_tokens_label = QLabel("Detected ASL Tokens:")
        self.english_tokens_label.setObjectName("sectionLabel")
        self.english_text_input = QLineEdit()
        self.english_text_input.setPlaceholderText("Type English text instead of speaking")
        self.english_text_input.installEventFilter(self)
        self.english_text_input.returnPressed.connect(self.on_typed_text_submit)
        self.english_text_submit_button = QPushButton("Translate Text")
        self.english_text_submit_button.setObjectName("secondaryButton")
        self.english_text_submit_button.setMinimumHeight(self.mini_button_height)
        self.english_text_submit_button.clicked.connect(self.on_typed_text_submit)

        self.record_button = QPushButton("●")
        self.record_button.setObjectName("recordToggleButton")
        record_size = 46 if self.compact_ui else 56
        self.record_button.setFixedSize(record_size, record_size)
        self.record_button.clicked.connect(self.on_record_clicked)
        self.record_button.setEnabled(False)
        self._refresh_record_toggle_button()
        self.replay_button = QPushButton("↻")
        self.replay_button.setObjectName("replayIconButton")
        self.replay_button.setFixedSize(record_size, record_size)
        self.replay_button.clicked.connect(self.on_replay_clicked)
        self.replay_button.setEnabled(False)
        self.replay_button.setToolTip("Replay last animation")
        self.favorite_button = QPushButton("☆")
        self.favorite_button.setObjectName("favoriteStarButton")
        self.favorite_button.setFixedSize(record_size, record_size)
        self.favorite_button.clicked.connect(self.on_add_favorite_clicked)
        self.favorite_button.setEnabled(False)
        self.favorite_button.setToolTip("Add to favorites")
        self.clear_english_history_button = QPushButton("Clear History")
        self.clear_english_history_button.setObjectName("secondaryButton")
        self.clear_english_history_button.setMinimumHeight(self.mini_button_height)
        self.clear_english_history_button.clicked.connect(self.on_clear_english_history_clicked)

        self.english_history_list = QListWidget()
        self.english_history_list.setMinimumHeight(48 if self.compact_ui else 64)
        self.english_history_list.setMaximumHeight(72 if self.compact_ui else 96)
        self.english_history_list.itemClicked.connect(self.on_english_history_item_clicked)
        self.english_history_list.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d1d1d6; border-radius: 10px; color: #1c1c1e;"
        )
        self.english_favorites_list = QListWidget()
        self.english_favorites_list.setMinimumHeight(48 if self.compact_ui else 64)
        self.english_favorites_list.setMaximumHeight(72 if self.compact_ui else 96)
        self.english_favorites_list.itemClicked.connect(self.on_english_favorite_item_clicked)
        self.english_favorites_list.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d1d1d6; border-radius: 10px; color: #1c1c1e;"
        )
        self.english_saved_tabs = QTabWidget()
        self.english_saved_tabs.setObjectName("englishSavedTabs")
        history_tab = QWidget()
        history_layout = QVBoxLayout()
        history_layout.setContentsMargins(6, 6, 6, 6)
        history_layout.addWidget(self.english_history_list)
        history_tab.setLayout(history_layout)
        favorites_tab = QWidget()
        favorites_layout = QVBoxLayout()
        favorites_layout.setContentsMargins(6, 6, 6, 6)
        favorites_layout.addWidget(self.english_favorites_list)
        favorites_tab.setLayout(favorites_layout)
        self.english_saved_tabs.addTab(history_tab, "History")
        self.english_saved_tabs.addTab(favorites_tab, "Favorites")
        self.english_saved_drawer = QWidget()
        self.english_saved_drawer.setObjectName("savedDrawerPanel")
        self.english_saved_drawer.setMinimumWidth(0)
        self.english_saved_drawer.setMaximumWidth(0)
        self.english_saved_drawer.setMaximumHeight(self.english_saved_drawer_max_height)
        self.english_saved_drawer.setVisible(False)
        drawer_layout = QVBoxLayout()
        drawer_layout.setContentsMargins(0, 0, 0, 0)
        drawer_layout.setSpacing(0)
        drawer_layout.addWidget(self.english_saved_tabs)
        self.english_saved_drawer.setLayout(drawer_layout)
        self.english_saved_drawer_toggle = QPushButton("Saved ◂")
        self.english_saved_drawer_toggle.setObjectName("drawerHandleButton")
        self.english_saved_drawer_toggle.setMinimumHeight(self.secondary_button_height)
        self.english_saved_drawer_toggle.clicked.connect(self.toggle_english_saved_drawer)
        self.english_saved_drawer_toggle.setToolTip("Show saved drawer")
        self.english_saved_drawer_anim = QPropertyAnimation(
            self.english_saved_drawer,
            b"maximumWidth",
            self,
        )
        self.english_saved_drawer_anim.setDuration(220)
        self.english_saved_drawer_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self.english_saved_drawer_anim.valueChanged.connect(self._on_english_drawer_anim_value)
        self.english_saved_drawer_anim.finished.connect(self._on_english_drawer_anim_finished)

        controls_panel = QWidget()
        controls_panel.setObjectName("bottomPanel")
        controls_panel.setMaximumHeight(self.english_controls_collapsed_height)
        self.english_controls_panel = controls_panel
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(6 if self.compact_ui else 8)
        status_row = QHBoxLayout()
        status_row.setSpacing(8 if self.compact_ui else 10)
        status_row.addWidget(self.english_status_label, 0)
        status_row.addWidget(self.english_tokens_label, 1)
        controls_layout.addLayout(status_row)
        typed_row = QHBoxLayout()
        typed_row.setSpacing(8 if self.compact_ui else 10)
        typed_row.addWidget(self.english_text_input, 1)
        typed_row.addWidget(self.english_text_submit_button, 0)
        controls_layout.addLayout(typed_row)
        controls_row = QHBoxLayout()
        controls_row.setSpacing(6 if self.compact_ui else 8)
        controls_row.addStretch(1)
        controls_row.addWidget(self.record_button, 0, Qt.AlignCenter)
        controls_row.addWidget(self.replay_button, 0, Qt.AlignCenter)
        controls_row.addWidget(self.favorite_button, 0, Qt.AlignCenter)
        controls_row.addWidget(self.clear_english_history_button)
        controls_row.addWidget(self.english_saved_drawer_toggle)
        controls_row.addStretch(1)
        controls_layout.addLayout(controls_row)
        drawer_row = QHBoxLayout()
        drawer_row.setSpacing(8 if self.compact_ui else 10)
        drawer_row.addStretch(1)
        drawer_row.addWidget(self.english_saved_drawer)
        drawer_row.addStretch(1)
        controls_layout.addLayout(drawer_row)
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
        self.camera_feed_label.setObjectName("previewSurface")
        self.camera_feed_label.setMinimumHeight(self.asl_preview_min_height)
        self.camera_feed_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.camera_feed_label.setAlignment(Qt.AlignCenter)
        self.reverse_status_label = QLabel("Status: Camera listening...")
        self.reverse_status_label.setObjectName("statusLine")
        self.reverse_debug_label = QLabel("Debug Match: (none) | conf=0.00 | streak=0")
        self.reverse_debug_label.setObjectName("debugLine")
        self.camera_label = QLabel("Detected ASL Tokens:")
        self.camera_label.setObjectName("statusLine")
        self.reverse_label = QLabel("English Translation:")
        self.reverse_label.setWordWrap(True)
        self.reverse_label.setObjectName("sectionLabel")
        self.camera_toggle_button = QPushButton("Stop Camera")
        self.camera_toggle_button.setObjectName("secondaryButton")
        self.camera_toggle_button.setMinimumHeight(self.secondary_button_height)
        self.camera_toggle_button.clicked.connect(self.toggle_camera)
        self.reset_translation_button = QPushButton("Reset Translation")
        self.reset_translation_button.setObjectName("secondaryButton")
        self.reset_translation_button.setMinimumHeight(self.secondary_button_height)
        self.reset_translation_button.clicked.connect(self.reset_translation)
        self.asl_settings_button = QPushButton("⚙")
        self.asl_settings_button.setObjectName("secondaryButton")
        gear_size = 34 if self.compact_ui else 40
        self.asl_settings_button.setFixedSize(gear_size, gear_size)
        self.asl_settings_button.setToolTip("ASL settings")
        self.asl_settings_button.clicked.connect(self._show_asl_settings_menu)

        self.asl_settings_menu = QMenu(self)
        self.asl_use_video_action = self.asl_settings_menu.addAction("Use Video File")
        self.asl_use_video_action.triggered.connect(self.import_asl_video_file)
        self.asl_use_live_camera_action = self.asl_settings_menu.addAction("Use Live Camera")
        self.asl_use_live_camera_action.triggered.connect(self.use_live_camera_input)
        self.asl_settings_menu.addSeparator()
        self.asl_reset_action = self.asl_settings_menu.addAction("Reset Translation")
        self.asl_reset_action.triggered.connect(self.reset_translation)
        self.asl_add_history_action = self.asl_settings_menu.addAction("Add to History")
        self.asl_add_history_action.triggered.connect(self.add_current_translation_to_history)
        self.asl_replay_latest_history_action = self.asl_settings_menu.addAction(
            "Replay Latest History"
        )
        self.asl_replay_latest_history_action.triggered.connect(
            self.on_replay_latest_reverse_history_clicked
        )
        self.asl_clear_history_action = self.asl_settings_menu.addAction("Clear ASL History")
        self.asl_clear_history_action.triggered.connect(self.on_clear_reverse_history_clicked)
        self.asl_settings_menu.addSeparator()
        self.asl_toggle_speaker_action = self.asl_settings_menu.addAction("")
        self.asl_toggle_speaker_action.triggered.connect(self.toggle_speaker)
        self.asl_test_speaker_action = self.asl_settings_menu.addAction("Test Speaker")
        self.asl_test_speaker_action.triggered.connect(self.test_speaker)
        self._refresh_asl_settings_actions()

        # Keep this list for existing history functionality, but move controls into gear menu.
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.on_history_item_clicked)

        controls_panel = QWidget()
        controls_panel.setObjectName("bottomPanel")
        controls_panel.setMaximumHeight(self.reverse_controls_max_height)
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(12, 8, 12, 8)
        controls_layout.setSpacing(6 if self.compact_ui else 8)
        controls_layout.addWidget(self.reverse_status_label)
        controls_layout.addWidget(self.camera_label)
        controls_layout.addWidget(self.reverse_label)

        actions_row = QHBoxLayout()
        actions_row.setSpacing(6 if self.compact_ui else 10)
        actions_row.addStretch(1)
        actions_row.addWidget(self.camera_toggle_button)
        actions_row.addWidget(self.reset_translation_button)
        actions_row.addWidget(self.asl_settings_button)
        actions_row.addStretch(1)
        controls_layout.addLayout(actions_row)

        controls_panel.setLayout(controls_layout)

        page_layout.addWidget(self.camera_feed_label, 1)
        page_layout.addWidget(controls_panel, 0)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8 if self.compact_ui else 12)
        page.setLayout(page_layout)
        return page

    def _current_asl_source_label(self) -> str:
        if self.asl_video_path:
            return Path(self.asl_video_path).name
        return "Live Camera"

    def _update_asl_source_buttons(self):
        if hasattr(self, "asl_use_live_camera_action"):
            self.asl_use_live_camera_action.setEnabled(self.asl_video_path is not None)

    def _show_asl_settings_menu(self):
        if not hasattr(self, "asl_settings_button") or not hasattr(self, "asl_settings_menu"):
            return
        self._refresh_asl_settings_actions()
        pos = self.asl_settings_button.mapToGlobal(
            self.asl_settings_button.rect().bottomLeft()
        )
        self.asl_settings_menu.exec(pos)

    def _refresh_asl_settings_actions(self):
        if not hasattr(self, "asl_toggle_speaker_action"):
            return
        self.tts_backend = self._resolve_tts_backend()
        speaker_text = "Disable Speaker" if self.speaker_enabled else "Enable Speaker"
        self.asl_toggle_speaker_action.setText(speaker_text)
        speaker_available = self.tts_backend != "none"
        self.asl_toggle_speaker_action.setEnabled(speaker_available)
        self.asl_test_speaker_action.setEnabled(speaker_available)
        has_history = hasattr(self, "history_list") and self.history_list.count() > 0
        self.asl_replay_latest_history_action.setEnabled(has_history)
        self.asl_clear_history_action.setEnabled(has_history)

    def start_camera(self, source_path: str | None = None):
        if self.camera_running:
            return
        if source_path is not None:
            self.asl_video_path = source_path
        self.camera_error_message = None
        self.pending_camera_tokens.clear()
        if hasattr(self, "camera_label"):
            self.camera_label.setText("Detected ASL Tokens:")
        if hasattr(self, "reverse_label"):
            self.reverse_label.setText("English Translation:")
        self.camera_thread = QThread(self)
        self.camera_worker = CameraWorker(
            source_path=self.asl_video_path,
            fast_mode=self.fast_camera_mode,
        )
        self.camera_worker.moveToThread(self.camera_thread)
        self.camera_thread.started.connect(self.camera_worker.run)
        self.camera_worker.pose_ready.connect(self.on_camera_pose)
        self.camera_worker.frame_ready.connect(self.camera_frame_received.emit)
        self.camera_worker.token_ready.connect(self.camera_token_received.emit)
        self.camera_worker.debug_ready.connect(self.camera_debug_received.emit)
        self.camera_worker.error_ready.connect(self.camera_error_received.emit)
        self.camera_worker.sequence_ready.connect(self.camera_sequence_received.emit)
        self.camera_worker.finished.connect(self.on_camera_worker_finished)
        self.camera_reset_requested.connect(self.camera_worker.reset_recognition_state)
        self.camera_stop_requested.connect(self.camera_worker.stop)
        self.camera_thread.start()
        self.camera_running = True
        self.camera_shutdown_in_progress = False
        source_name = self._current_asl_source_label()
        mode_suffix = " (Fast)" if self.fast_camera_mode else ""
        if self.asl_video_path:
            self.camera_state_label.setText(f"Video: Playing ({source_name}){mode_suffix}")
        else:
            self.camera_state_label.setText(f"Camera: Ready{mode_suffix}")
        if hasattr(self, "camera_toggle_button"):
            self.camera_toggle_button.setText("Stop Input")
        if hasattr(self, "reverse_status_label"):
            if self.asl_video_path:
                self.reverse_status_label.setText(f"Status: Processing video {source_name}{mode_suffix}")
            else:
                self.reverse_status_label.setText(f"Status: Camera listening...{mode_suffix}")
        if hasattr(self, "reverse_debug_label"):
            self.reverse_debug_label.setText("Debug Match: (none) | conf=0.00 | streak=0")
        if hasattr(self, "camera_feed_label"):
            self.camera_feed_label.setText("Video preview" if self.asl_video_path else "Camera feed")
            self.camera_feed_label.setPixmap(QPixmap())
        self._update_asl_source_buttons()

    def stop_camera(self, finalize_pending: bool = True):
        if not self.camera_running:
            return
        self.camera_shutdown_in_progress = True
        self.latest_camera_pose = None
        self.latest_camera_debug_token = ""
        self.latest_camera_debug_confidence = 0.0
        translated_on_stop = False
        if finalize_pending and self.pending_camera_tokens:
            self._finalize_camera_translation(list(self.pending_camera_tokens))
            translated_on_stop = True
        elif not finalize_pending:
            self.pending_camera_tokens.clear()
        worker = self.camera_worker
        thread = self.camera_thread
        if worker is not None:
            try:
                self.camera_stop_requested.emit()
            except Exception:
                pass
        if thread is not None:
            try:
                self.camera_reset_requested.disconnect(worker.reset_recognition_state)
            except Exception:
                pass
            try:
                self.camera_stop_requested.disconnect(worker.stop)
            except Exception:
                pass
            try:
                worker.finished.disconnect(self.on_camera_worker_finished)
            except Exception:
                pass
            thread.requestInterruption()
            thread.quit()
            thread.wait(2500)
        self.camera_thread = None
        self.camera_worker = None
        self.camera_running = False
        self.camera_shutdown_in_progress = False
        if self.asl_video_path:
            self.camera_state_label.setText("Video: Stopped")
        else:
            self.camera_state_label.setText("Camera: Stopped")
        if hasattr(self, "camera_toggle_button"):
            self.camera_toggle_button.setText("Start Input")
        if hasattr(self, "reverse_status_label") and not translated_on_stop:
            if self.camera_error_message:
                self.reverse_status_label.setText(
                    f"Status: Camera error: {self.camera_error_message}"
                )
            else:
                self.reverse_status_label.setText(
                    "Status: Video stopped" if self.asl_video_path else "Status: Camera stopped"
                )
        if hasattr(self, "reverse_debug_label"):
            self.reverse_debug_label.setText("Debug Match: (none) | conf=0.00 | streak=0")
        if hasattr(self, "camera_feed_label"):
            self.camera_feed_label.setPixmap(QPixmap())
            self.camera_feed_label.setText("Video stopped" if self.asl_video_path else "Camera stopped")
        self._update_asl_source_buttons()

    def toggle_camera(self):
        if self.camera_running:
            self.stop_camera()
        else:
            self.start_camera()

    def _refresh_fast_mode_button(self):
        if hasattr(self, "fast_mode_button"):
            self.fast_mode_button.setText(
                "Fast Mode: ON" if self.fast_camera_mode else "Fast Mode: OFF"
            )

    def toggle_fast_mode(self):
        self.fast_camera_mode = not self.fast_camera_mode
        self._refresh_fast_mode_button()
        source_name = self._current_asl_source_label()
        mode_name = "FAST" if self.fast_camera_mode else "NORMAL"
        self.reverse_status_label.setText(
            f"Status: Camera mode set to {mode_name} ({source_name})"
        )
        if self.camera_running and self.asl_video_path is None:
            source_path = self.asl_video_path
            self.stop_camera(finalize_pending=False)
            self.reset_translation()
            self.start_camera(source_path=source_path)

    def import_asl_video_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ASL Video",
            "",
            "Video Files (*.mp4 *.mov *.avi *.mkv *.m4v *.webm);;All Files (*)",
        )
        if not file_path:
            return
        was_running = self.camera_running
        if was_running:
            self.stop_camera(finalize_pending=False)
        self.reset_translation()
        self.start_camera(source_path=file_path)

    def use_live_camera_input(self):
        if self.asl_video_path is None and self.camera_running:
            return
        was_running = self.camera_running
        if was_running:
            self.stop_camera(finalize_pending=False)
        self.asl_video_path = None
        self.reset_translation()
        if self.mode == "asl_to_english":
            self.start_camera()

    def on_camera_worker_finished(self, source_is_file: bool):
        if self.camera_shutdown_in_progress or not self.camera_running:
            return
        self.latest_camera_pose = None
        self.latest_camera_debug_token = ""
        self.latest_camera_debug_confidence = 0.0
        if source_is_file and self.pending_camera_tokens:
            self._finalize_camera_translation(list(self.pending_camera_tokens))
        worker = self.camera_worker
        thread = self.camera_thread
        if worker is not None:
            try:
                self.camera_reset_requested.disconnect(worker.reset_recognition_state)
            except Exception:
                pass
            try:
                self.camera_stop_requested.disconnect(worker.stop)
            except Exception:
                pass
            try:
                worker.finished.disconnect(self.on_camera_worker_finished)
            except Exception:
                pass
        if thread is not None:
            thread.quit()
            thread.wait(2500)
        self.camera_thread = None
        self.camera_worker = None
        self.camera_running = False
        if source_is_file:
            self.camera_state_label.setText("Video: Complete")
            self.reverse_status_label.setText("Status: Video processing complete")
            self.camera_feed_label.setText("Video complete")
        else:
            self.camera_state_label.setText("Camera: Stopped")
            self.reverse_status_label.setText("Status: Camera stopped")
            self.camera_feed_label.setText("Camera stopped")
        self.camera_toggle_button.setText("Start Input")
        self.reverse_debug_label.setText("Debug Match: (none) | conf=0.00 | streak=0")
        self._update_asl_source_buttons()

    def reset_translation(self):
        self.pending_camera_tokens.clear()
        self.latest_camera_debug_token = ""
        self.latest_camera_debug_confidence = 0.0
        self.camera_label.setText("Detected ASL Tokens:")
        self.reverse_label.setText("English Translation:")
        self.latest_translation_text = ""
        self.reverse_debug_label.setText("Debug Match: (none) | conf=0.00 | streak=0")
        if self.camera_running:
            if self.asl_video_path:
                self.reverse_status_label.setText(
                    f"Status: Processing video {self._current_asl_source_label()}"
                )
            else:
                self.reverse_status_label.setText("Status: Camera listening...")
            self.camera_reset_requested.emit()
        else:
            self.reverse_status_label.setText(
                "Status: Video stopped" if self.asl_video_path else "Status: Camera stopped"
            )

    def add_current_translation_to_history(self):
        text = (self.latest_translation_text or "").strip()
        if not text:
            self.reverse_status_label.setText("Status: No translation to add")
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        self.history_list.insertItem(0, f"[{stamp}] {text}")
        self.reverse_status_label.setText("Status: Added to history")
        self._refresh_asl_settings_actions()

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
        if (
            shutil.which("espeak")
            or shutil.which("espeak-ng")
            or os.path.exists("/usr/bin/espeak")
            or os.path.exists("/usr/bin/espeak-ng")
        ):
            return "espeak"
        if self.tts_engine is not None:
            return "qt"
        return "none"

    def toggle_speaker(self):
        self.tts_backend = self._resolve_tts_backend()
        if self.tts_backend == "none":
            self.reverse_status_label.setText("Status: Speaker unavailable")
            self._refresh_asl_settings_actions()
            return
        self.speaker_enabled = not self.speaker_enabled
        if self.speaker_enabled:
            self.reverse_status_label.setText(
                f"Status: Speaker enabled ({self.tts_backend.upper()})"
            )
            self._speak_text("Speaker enabled")
        else:
            self.reverse_status_label.setText("Status: Speaker disabled")
        self._refresh_asl_settings_actions()

    def _speak_translation_if_enabled(self, text: str):
        if not self.speaker_enabled:
            return
        if not text:
            return
        self._speak_text(text)

    def _speak_text(self, text: str):
        if not text:
            return
        self._sync_linux_playback_volume(force=False)
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
            tts_cmd = (
                shutil.which("espeak-ng")
                or shutil.which("espeak")
                or ("/usr/bin/espeak-ng" if os.path.exists("/usr/bin/espeak-ng") else None)
                or ("/usr/bin/espeak" if os.path.exists("/usr/bin/espeak") else None)
            )
            if tts_cmd:
                try:
                    with tempfile.NamedTemporaryFile(
                        prefix="asl_tts_",
                        suffix=".wav",
                        delete=False,
                    ) as wavf:
                        wav_path = wavf.name
                    subprocess.run(
                        [
                            tts_cmd,
                            "-a",
                            str(self.espeak_amplitude),
                            "-s",
                            str(self.espeak_speed),
                            "-w",
                            wav_path,
                            text,
                        ],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    subprocess.Popen(
                        ["aplay", "-D", self.tts_alsa_device, wav_path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    subprocess.Popen(
                        ["sh", "-c", f"sleep 2; rm -f '{wav_path}'"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return
                except Exception:
                    pass
        self.reverse_status_label.setText("Status: Speaker error")

    def _sync_linux_playback_volume(self, force: bool = False) -> bool:
        if not sys.platform.startswith("linux"):
            return False
        now = time.monotonic()
        if not force and (now - self._last_volume_sync) < 8.0:
            return False
        amixer_cmd = shutil.which("amixer")
        if not amixer_cmd:
            return False
        target = f"{self.tts_output_volume}%"
        default_controls = ("PCM", "Master", "Speaker", "Headphone", "Playback")
        card_controls = ("Speaker", "Headphone", "Playback", "PCM")
        changed = False
        for control in default_controls:
            try:
                result = subprocess.run(
                    [amixer_cmd, "sset", control, target, "unmute"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                if result.returncode == 0:
                    changed = True
            except Exception:
                continue
        if self.tts_mixer_card:
            for control in card_controls:
                try:
                    result = subprocess.run(
                        [amixer_cmd, "-c", self.tts_mixer_card, "sset", control, target, "unmute"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                    )
                    if result.returncode == 0:
                        changed = True
                except Exception:
                    continue
        self._last_volume_sync = now
        return changed

    def test_speaker(self):
        self.tts_backend = self._resolve_tts_backend()
        if self.tts_backend == "none":
            self.reverse_status_label.setText("Status: Speaker unavailable")
            return
        self._sync_linux_playback_volume(force=True)
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
        set_demo_clip_mode(self.demo_mode)
        using_demo_clips = is_demo_clip_source_active()
        self.demo_mode_button.setProperty("enabledState", self.demo_mode)
        if self.demo_mode:
            self.demo_mode_button.setText("Demo Mode: ON")
            self.reverse_debug_label.hide()
            vocab_hint = f"Demo ASL vocab: {len(self.demo_allowed_tokens)} words"
            if using_demo_clips:
                self.demo_mode_button.setToolTip(
                    f"{vocab_hint}. Using curated clips from: {DEMO_CLIP_DIR}"
                )
            else:
                self.demo_mode_button.setToolTip(
                    f"{vocab_hint}. No curated clips found in: {DEMO_CLIP_DIR}. Using default clips."
                )
        else:
            self.demo_mode_button.setText("Demo Mode: OFF")
            self.reverse_debug_label.show()
            self.demo_mode_button.setToolTip("Using default clip library")
        self.demo_mode_button.style().unpolish(self.demo_mode_button)
        self.demo_mode_button.style().polish(self.demo_mode_button)

    def refresh_battery_status(self):
        status = read_battery_status()
        self.battery_state_label.setText(status.label)

    def _detect_compact_ui(self) -> bool:
        touch_env = os.getenv("ASL_TOUCH_UI", "").strip().lower()
        if touch_env in {"1", "true", "yes", "on"}:
            return True
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return False
        geo = screen.availableGeometry()
        return geo.width() <= 1024 or geo.height() <= 600

    def _detect_portrait_ui(self) -> bool:
        portrait_env = os.getenv("ASL_PORTRAIT_UI", "").strip().lower()
        if portrait_env in {"1", "true", "yes", "on"}:
            return True
        if portrait_env in {"0", "false", "no", "off"}:
            return False
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return False
        geo = screen.availableGeometry()
        return geo.height() > geo.width()

    def _apply_theme(self):
        mode_font = 14 if self.compact_ui else 16
        mode_height = 36 if self.compact_ui else 44
        demo_height = self.mini_button_height
        css = """
            QWidget {
                background-color: #eef2f7;
                color: #111827;
                font-family: "SF Pro Text", ".AppleSystemUIFont", "Helvetica Neue", sans-serif;
            }
            QLabel {
                color: #111827;
                background: transparent;
            }
            QStackedWidget {
                background-color: transparent;
                border: none;
            }
            QWidget#topPanel,
            QWidget#bottomPanel,
            QWidget#savedDrawerPanel {
                background-color: #fbfcfe;
                border: 1px solid #d8e0ea;
                border-radius: 20px;
            }
            QWidget#bottomPanel {
                border-top-left-radius: 20px;
                border-top-right-radius: 20px;
                border-bottom-left-radius: 0px;
                border-bottom-right-radius: 0px;
            }
            QWidget#modeShell {
                background-color: #e9eef5;
                border: 1px solid #d8e0ea;
                border-radius: 18px;
            }
            QLabel#pageTitle {
                font-size: 24px;
                font-weight: 700;
                letter-spacing: 0.2px;
            }
            QLabel#pageSubtitle {
                color: #6b7280;
                font-size: 13px;
            }
            QLabel#modeBadge {
                color: #0b63ce;
                background-color: #e7f0ff;
                border: 1px solid #c8dafc;
                border-radius: 12px;
                padding: 6px 10px;
                font-size: 12px;
                font-weight: 700;
            }
            QLabel#statusChip {
                color: #334155;
                background-color: #f4f7fb;
                border: 1px solid #d8e0ea;
                border-radius: 11px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: 600;
            }
            QLabel#statusLine {
                font-size: 13px;
                color: #475569;
                font-weight: 600;
            }
            QLabel#debugLine {
                font-size: 12px;
                color: #94a3b8;
            }
            QLabel#heroValue {
                font-size: 19px;
                font-weight: 700;
                color: #0f172a;
                padding-top: 2px;
            }
            QLabel#sectionLabel {
                font-size: 13px;
                font-weight: 700;
                color: #475569;
            }
            QLabel#previewSurface,
            QWidget#previewSurface {
                background-color: #f8fafc;
                border: 1px solid #d8e0ea;
                border-radius: 24px;
                color: #94a3b8;
            }
            QLineEdit {
                background-color: #ffffff;
                border: 1px solid #d8e0ea;
                border-radius: 10px;
                color: #0f172a;
                padding: 6px 10px;
                selection-background-color: #cfe3ff;
                selection-color: #0f172a;
            }
            QLineEdit:focus {
                border: 1px solid #93c5fd;
                background-color: #ffffff;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d8e0ea;
                border-radius: 14px;
                color: #0f172a;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background-color: #f8fafc;
                border-color: #cbd5e1;
            }
            QPushButton:pressed {
                background-color: #edf2f7;
            }
            QPushButton#modeButton {
                font-size: __MODE_FONT__px;
                font-weight: 700;
                min-height: __MODE_HEIGHT__px;
                border-radius: 12px;
                border-color: transparent;
                background-color: transparent;
                color: #64748b;
            }
            QPushButton#modeButton[active="true"] {
                background-color: #ffffff;
                border-color: #d8e0ea;
                color: #0b63ce;
            }
            QPushButton#recordToggleButton {
                background-color: #ff5a52;
                border: 2px solid #ffd0cc;
                border-radius: 999px;
                color: #ffffff;
                font-weight: 800;
                font-size: 24px;
                padding: 0px;
            }
            QPushButton#recordToggleButton:hover {
                background-color: #ff6b63;
                border-color: #ffe3e0;
            }
            QPushButton#recordToggleButton:pressed {
                background-color: #ef4d45;
            }
            QPushButton#recordToggleButton[recording="true"] {
                background-color: #ff5a52;
                border-color: #ffe7e5;
                border-radius: 14px;
            }
            QPushButton#recordToggleButton:disabled {
                background-color: #f5b7b3;
                border-color: #edc1be;
                color: #ffffff;
            }
            QPushButton#replayIconButton,
            QPushButton#favoriteStarButton {
                background-color: #ffffff;
                border: 1px solid #d8e0ea;
                border-radius: 999px;
                padding: 0px;
            }
            QPushButton#replayIconButton {
                color: #64748b;
                font-weight: 800;
                font-size: 22px;
            }
            QPushButton#favoriteStarButton {
                color: #94a3b8;
                font-weight: 800;
                font-size: 24px;
            }
            QPushButton#replayIconButton:hover,
            QPushButton#favoriteStarButton:hover {
                background-color: #f8fafc;
            }
            QPushButton#replayIconButton:pressed,
            QPushButton#favoriteStarButton:pressed {
                background-color: #edf2f7;
            }
            QPushButton#replayIconButton:disabled,
            QPushButton#favoriteStarButton:disabled {
                background-color: #f8fafc;
                color: #cbd5e1;
                border-color: #e2e8f0;
            }
            QPushButton#favoriteStarButton[favorited="true"] {
                background-color: #fff8e7;
                border-color: #f7d774;
                color: #eab308;
            }
            QPushButton#drawerHandleButton,
            QPushButton#secondaryButton {
                background-color: #ffffff;
                border-color: #d8e0ea;
                color: #334155;
                font-weight: 600;
            }
            QPushButton#drawerHandleButton {
                color: #0b63ce;
                min-width: 86px;
                padding: 6px 12px;
            }
            QPushButton#drawerHandleButton:hover,
            QPushButton#secondaryButton:hover {
                background-color: #f8fafc;
            }
            QPushButton#drawerHandleButton:pressed,
            QPushButton#secondaryButton:pressed {
                background-color: #edf2f7;
            }
            QPushButton#demoButton {
                min-height: __DEMO_HEIGHT__px;
                border-radius: 12px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: 700;
            }
            QPushButton#demoButton[enabledState="true"] {
                background-color: #34c759;
                border-color: #2aa44b;
                color: #ffffff;
            }
            QPushButton#demoButton[enabledState="false"] {
                background-color: #94a3b8;
                border-color: #7b8aa1;
                color: #ffffff;
            }
            QPushButton#closeButton {
                background-color: #ffffff;
                border: 1px solid #d8e0ea;
                border-radius: 999px;
                color: #475569;
                font-size: 16px;
                font-weight: 800;
                padding: 0px;
            }
            QPushButton#closeButton:hover {
                background-color: #fee2e2;
                border-color: #fca5a5;
                color: #b91c1c;
            }
            QPushButton#closeButton:pressed {
                background-color: #fecaca;
                border-color: #f87171;
                color: #991b1b;
            }
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #d8e0ea;
                border-radius: 14px;
                color: #111827;
                padding: 4px;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-radius: 10px;
            }
            QListWidget::item:selected {
                background-color: #e7f0ff;
                color: #0f172a;
            }
            QTabWidget::pane {
                border: 1px solid #d8e0ea;
                border-radius: 14px;
                background: #ffffff;
                top: -1px;
            }
            QTabBar::tab {
                background: #eef2f7;
                border: 1px solid #d8e0ea;
                border-bottom: none;
                padding: 6px 14px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                color: #64748b;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #ffffff;
                color: #0b63ce;
                border-color: #d8e0ea;
            }
            """
        css = css.replace("__MODE_FONT__", str(mode_font))
        css = css.replace("__MODE_HEIGHT__", str(mode_height))
        css = css.replace("__DEMO_HEIGHT__", str(demo_height))
        self.setStyleSheet(css)
