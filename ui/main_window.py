from core.engine import english_to_asl, asl_to_english, _get_stt_backend
from core.mic_utils import record_audio_until_stop
from core.audio.vosk_listener import VoskListener
import json
import threading
import subprocess
import sys
import shutil
import os
import tempfile
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import QThread, Qt, QTimer, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QGuiApplication
from core.sequencing.sign_sequencer import sequence_signs, SignEvent
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
    QTabWidget,
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
        self.record_stop_event = None
        self.recording_in_progress = False
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
        self.pending_camera_tokens = []
        self.camera_error_message = None
        self.demo_mode = True
        self.latest_translation_text = ""
        self.tts_engine = self._init_tts_engine()
        self.tts_backend = self._resolve_tts_backend()
        self.espeak_amplitude = _env_int("ASL_TTS_AMPLITUDE", 180, min_value=0, max_value=200)
        self.espeak_speed = _env_int("ASL_TTS_SPEED", 160, min_value=80, max_value=300)
        self.tts_alsa_device = os.getenv(
            "ASL_TTS_ALSA_DEVICE",
            "default:CARD=wm8960soundcard",
        )
        self.speaker_enabled = self.tts_backend != "none"
        self.compact_ui = self._detect_compact_ui()
        self.english_drawer_width = 300 if self.compact_ui else 420
        self.preview_min_height = 360 if self.compact_ui else 520
        self.english_preview_min_height = 520 if self.compact_ui else 780
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
            f"font-size: {self.small_font}px; font-weight: 700; color: #005ecb; "
            "background-color: #e8f2ff; border: 1px solid #c7dcff; "
            "border-radius: 10px; padding: 6px 10px;"
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
            "background-color: #34c759; color: #ffffff; border: 1px solid #2ba84a; border-radius: 10px;"
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
                f"font-size: {self.small_font}px; color: #3a3a3c; "
                "background-color: #ffffff; border: 1px solid #d1d1d6; border-radius: 9px; padding: 5px 9px;"
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

        self._load_english_collections()
        self._refresh_english_collections_ui()

        self.start_camera()

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
        self.record_stop_event = threading.Event()
        self._refresh_record_toggle_button()

        if getattr(self, "vosk", None) is not None:
            self.vosk.stop()

        threading.Thread(target=self._run_record_job, daemon=True).start()

    def on_stop_record_clicked(self):
        if not self.recording_in_progress:
            return
        self.english_status_label.setText("Status: Stopping recording...")
        self.recording_in_progress = False
        self.record_button.setEnabled(False)
        self._refresh_record_toggle_button()
        if self.record_stop_event is not None:
            self.record_stop_event.set()

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

        if self.record_stop_event is not None:
            self.record_stop_event.set()

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
        if not self.english_drawer_open:
            self.english_saved_drawer.setVisible(True)
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
            stop_event = self.record_stop_event
            if stop_event is None:
                raise RuntimeError("Recording session was not initialized.")
            audio = record_audio_until_stop(stop_event=stop_event)
            print(f"[Record] captured bytes={len(audio)}")
            if not audio:
                raise RuntimeError("No audio captured. Try recording again.")
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
        self.stt_ready = ok
        self.record_button.setEnabled(ok)
        self._refresh_record_toggle_button()
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
        self.english_animation_view.setMinimumHeight(self.english_preview_min_height)
        self.english_animation_view.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.english_status_label = QLabel("Status: Idle")
        self.english_status_label.setStyleSheet(f"font-size: {self.body_font}px;")
        self.english_tokens_label = QLabel("Detected ASL Tokens:")
        self.english_tokens_label.setStyleSheet(f"font-size: {self.heading_font}px;")

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
        self.remove_favorite_button = QPushButton("Remove Favorite")
        self.remove_favorite_button.setObjectName("secondaryButton")
        self.remove_favorite_button.setMinimumHeight(self.secondary_button_height)
        self.remove_favorite_button.clicked.connect(self.on_remove_favorite_clicked)
        self.clear_english_history_button = QPushButton("Clear History")
        self.clear_english_history_button.setObjectName("secondaryButton")
        self.clear_english_history_button.setMinimumHeight(self.secondary_button_height)
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
        self.english_saved_drawer.setMaximumHeight(96 if self.compact_ui else 128)
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
        controls_panel.setMaximumHeight(220 if self.compact_ui else 270)
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(8, 6, 8, 6)
        controls_layout.setSpacing(6 if self.compact_ui else 8)
        controls_layout.addWidget(self.english_status_label)
        controls_layout.addWidget(self.english_tokens_label)
        controls_layout.addStretch(1)
        controls_row = QHBoxLayout()
        controls_row.setSpacing(8 if self.compact_ui else 10)
        controls_row.addStretch(1)
        controls_row.addWidget(self.record_button, 0, Qt.AlignCenter)
        controls_row.addWidget(self.replay_button, 0, Qt.AlignCenter)
        controls_row.addWidget(self.favorite_button, 0, Qt.AlignCenter)
        controls_row.addStretch(1)
        controls_layout.addLayout(controls_row)
        quick_row = QHBoxLayout()
        quick_row.setSpacing(8 if self.compact_ui else 10)
        quick_row.addStretch(1)
        quick_row.addWidget(self.clear_english_history_button)
        controls_layout.addLayout(quick_row)
        drawer_row = QHBoxLayout()
        drawer_row.setSpacing(8 if self.compact_ui else 10)
        drawer_row.addStretch(1)
        drawer_row.addWidget(self.english_saved_drawer_toggle, 0, Qt.AlignRight)
        drawer_row.addWidget(self.english_saved_drawer)
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
        self.camera_feed_label.setMinimumHeight(self.preview_min_height)
        self.camera_feed_label.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.camera_feed_label.setAlignment(Qt.AlignCenter)
        self.camera_feed_label.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d1d1d6; "
            f"border-radius: 14px; font-size: {self.body_font}px; color: #6b7280;"
        )
        self.reverse_status_label = QLabel("Status: Camera listening...")
        self.reverse_status_label.setStyleSheet(f"font-size: {self.body_font}px;")
        self.reverse_debug_label = QLabel("Debug Match: (none) | conf=0.00 | streak=0")
        self.reverse_debug_label.setStyleSheet(f"font-size: {self.small_font}px; color: #8e8e93;")
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
            self.speaker_button.setText("Speaker: OFF")

        self.history_label = QLabel("History")
        self.history_label.setStyleSheet(f"font-size: {self.body_font}px; font-weight: 700;")
        self.history_list = QListWidget()
        self.history_list.setMinimumHeight(52 if self.compact_ui else 70)
        self.history_list.setMaximumHeight(78 if self.compact_ui else 96)
        self.history_list.itemClicked.connect(self.on_history_item_clicked)
        self.history_list.setStyleSheet(
            "background-color: #ffffff; border: 1px solid #d1d1d6; border-radius: 10px; color: #1c1c1e;"
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

    def test_speaker(self):
        self.tts_backend = self._resolve_tts_backend()
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
                "background-color: #34c759; color: #ffffff; border: 1px solid #2ba84a; border-radius: 10px;"
            )
            self.reverse_debug_label.hide()
        else:
            self.demo_mode_button.setText("Demo Mode: OFF")
            self.demo_mode_button.setStyleSheet(
                f"font-size: {self.small_font}px; font-weight: 700; height: {self.mini_button_height}px; "
                "background-color: #8e8e93; color: #ffffff; border: 1px solid #7d7d82; border-radius: 10px;"
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
                background-color: #f2f2f7;
                color: #1c1c1e;
                font-family: "SF Pro Text", ".AppleSystemUIFont", "Helvetica Neue", sans-serif;
            }
            QLabel {
                color: #1c1c1e;
            }
            QStackedWidget {
                background-color: transparent;
                border: none;
            }
            QPushButton {
                background-color: #ffffff;
                border: 1px solid #d1d1d6;
                border-radius: 10px;
                color: #1c1c1e;
                padding: 8px 14px;
            }
            QPushButton:hover {
                background-color: #f7f7fa;
                border-color: #c8c7cc;
            }
            QPushButton:pressed {
                background-color: #ececf1;
            }
            QPushButton#modeButton {
                font-size: __MODE_FONT__px;
                font-weight: 700;
                min-height: __MODE_HEIGHT__px;
            }
            QPushButton#modeButton[active="true"] {
                background-color: #007aff;
                border-color: #0062cc;
                color: #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryButton {
                background-color: #007aff;
                border-color: #0062cc;
                color: #ffffff;
                font-weight: 700;
            }
            QPushButton#primaryButton:hover {
                background-color: #1a86ff;
            }
            QPushButton#primaryButton:pressed {
                background-color: #0066d6;
            }
            QPushButton#recordToggleButton {
                background-color: #ff3b30;
                border: 2px solid #ffb1ac;
                border-radius: 999px;
                color: #ffffff;
                font-weight: 800;
                font-size: 24px;
                padding: 0px;
            }
            QPushButton#recordToggleButton:hover {
                background-color: #ff5148;
                border-color: #ffcbc7;
            }
            QPushButton#recordToggleButton:pressed {
                background-color: #e2342a;
            }
            QPushButton#recordToggleButton[recording="true"] {
                background-color: #ff3b30;
                border-color: #ffd8d5;
                border-radius: 8px;
            }
            QPushButton#recordToggleButton:disabled {
                background-color: #f3b4b1;
                border-color: #e8a2a0;
                color: #ffffff;
            }
            QPushButton#replayIconButton {
                background-color: #ffffff;
                border: 1px solid #d1d1d6;
                border-radius: 999px;
                color: #6e6e73;
                font-weight: 800;
                font-size: 22px;
                padding: 0px;
            }
            QPushButton#replayIconButton:hover {
                background-color: #f2f2f7;
            }
            QPushButton#replayIconButton:pressed {
                background-color: #e9e9ee;
            }
            QPushButton#replayIconButton:disabled {
                background-color: #f7f7fa;
                color: #b0b0b6;
            }
            QPushButton#favoriteStarButton {
                background-color: #ffffff;
                border: 1px solid #d1d1d6;
                border-radius: 999px;
                color: #8e8e93;
                font-weight: 800;
                font-size: 24px;
                padding: 0px;
            }
            QPushButton#favoriteStarButton:hover {
                background-color: #f2f2f7;
            }
            QPushButton#favoriteStarButton[favorited="true"] {
                background-color: #fff9e6;
                border-color: #f2c94c;
                color: #ffcc00;
            }
            QPushButton#favoriteStarButton:disabled {
                background-color: #f7f7fa;
                color: #b0b0b6;
                border-color: #e2e2e8;
            }
            QPushButton#drawerHandleButton {
                background-color: #ffffff;
                border: 1px solid #c7d7ef;
                border-radius: 10px;
                color: #007aff;
                font-weight: 700;
                min-width: 86px;
                padding: 6px 10px;
            }
            QPushButton#drawerHandleButton:hover {
                background-color: #f2f8ff;
            }
            QWidget#savedDrawerPanel {
                background-color: #ffffff;
                border: 1px solid #d1d1d6;
                border-radius: 10px;
            }
            QPushButton#secondaryButton {
                background-color: #ffffff;
                border-color: #d1d1d6;
                color: #1c1c1e;
                font-weight: 600;
            }
            QPushButton#secondaryButton:hover {
                background-color: #f2f2f7;
            }
            QPushButton#demoButton {
                border-radius: 10px;
                min-height: __DEMO_HEIGHT__px;
            }
            QWidget#bottomPanel {
                background-color: #ffffff;
                border: 1px solid #d1d1d6;
                border-radius: 14px;
            }
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #d1d1d6;
                border-radius: 10px;
                color: #1c1c1e;
            }
            QListWidget::item:selected {
                background-color: #dbeafe;
                color: #0f172a;
            }
            QTabWidget::pane {
                border: 1px solid #d1d1d6;
                border-radius: 10px;
                background: #ffffff;
            }
            QTabBar::tab {
                background: #f2f2f7;
                border: 1px solid #d1d1d6;
                border-bottom: none;
                padding: 5px 12px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                color: #3a3a3c;
            }
            QTabBar::tab:selected {
                background: #007aff;
                color: #ffffff;
                border-color: #0062cc;
            }
            """
        css = css.replace("__MODE_FONT__", str(mode_font))
        css = css.replace("__MODE_HEIGHT__", str(mode_height))
        css = css.replace("__DEMO_HEIGHT__", str(demo_height))
        self.setStyleSheet(css)
