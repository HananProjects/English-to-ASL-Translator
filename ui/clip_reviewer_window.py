import json
import re
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.english_to_asl.dictionary.asl_signs import ASL_SIGNS
from core.sequencing.sign_sequencer import SignEvent
from ui.widgets.animation_view import ASLAnimationView


def _clip_to_tokens() -> dict[str, list[str]]:
    mapping: dict[str, set[str]] = {}
    for token, definition in ASL_SIGNS.items():
        clip_name = definition.get("clip")
        if isinstance(clip_name, str) and clip_name:
            mapping.setdefault(clip_name, set()).add(token)
        clip_list = definition.get("clips")
        if isinstance(clip_list, list):
            for item in clip_list:
                if isinstance(item, str) and item:
                    mapping.setdefault(item, set()).add(token)
    return {k: sorted(v) for k, v in mapping.items()}


def _base_stem(stem: str) -> str:
    # about_2 -> about, no_1_2 -> no
    return re.sub(r"(?:_\d+)+$", "", stem)


class ClipReviewerWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.repo_root = Path(__file__).resolve().parents[1]
        self.clip_dir = self.repo_root / "ui" / "animation" / "clips"
        self.bad_dir = self.clip_dir / "_bad"
        self.clip_token_map = _clip_to_tokens()
        self.all_clips: list[Path] = []

        self.setWindowTitle("ASL Clip Reviewer")
        self.setMinimumSize(1100, 680)
        self.resize(1280, 820)

        root = QHBoxLayout()
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(12)

        left_panel = QVBoxLayout()
        left_panel.setSpacing(8)
        left_panel.addWidget(QLabel("Clip files (.json)"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter clips (name)")
        self.search_input.textChanged.connect(self._refresh_list)
        left_panel.addWidget(self.search_input)

        self.clip_list = QListWidget()
        self.clip_list.itemSelectionChanged.connect(self._on_selected_clip_changed)
        left_panel.addWidget(self.clip_list, 1)

        list_actions = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.reload_button = QPushButton("Reload")
        self.prev_button.clicked.connect(self._select_previous)
        self.next_button.clicked.connect(self._select_next)
        self.reload_button.clicked.connect(self._reload_clips)
        list_actions.addWidget(self.prev_button)
        list_actions.addWidget(self.next_button)
        list_actions.addWidget(self.reload_button)
        left_panel.addLayout(list_actions)

        left_host = QWidget()
        left_host.setLayout(left_panel)
        left_host.setMinimumWidth(360)
        root.addWidget(left_host, 0)

        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)
        header = QLabel("Avatar preview")
        header.setStyleSheet("font-size: 22px; font-weight: 700;")
        right_panel.addWidget(header)

        self.preview = ASLAnimationView()
        self.preview.disable_live_pose()
        self.preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.addWidget(self.preview, 1)

        self.clip_name_label = QLabel("Clip: (none)")
        self.prints_label = QLabel("Avatar prints: (none)")
        self.meta_label = QLabel("Frames: - | FPS: - | Duration: - | Size: -")
        self.tokens_label = QLabel("Dictionary tokens: (none)")
        for label in (
            self.clip_name_label,
            self.prints_label,
            self.meta_label,
            self.tokens_label,
        ):
            label.setWordWrap(True)
            right_panel.addWidget(label)

        action_row = QHBoxLayout()
        self.play_button = QPushButton("Play Clip")
        self.move_bad_button = QPushButton("Move to _bad")
        self.delete_button = QPushButton("Delete Permanently")
        self.play_button.clicked.connect(self._play_selected_clip)
        self.move_bad_button.clicked.connect(self._move_selected_to_bad)
        self.delete_button.clicked.connect(self._delete_selected_clip)
        action_row.addWidget(self.play_button)
        action_row.addWidget(self.move_bad_button)
        action_row.addWidget(self.delete_button)
        right_panel.addLayout(action_row)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #475569;")
        right_panel.addWidget(self.status_label)

        right_host = QWidget()
        right_host.setLayout(right_panel)
        root.addWidget(right_host, 1)

        self.setLayout(root)
        self._apply_theme()
        self._reload_clips()

    def _apply_theme(self):
        self.setStyleSheet(
            """
            QWidget {
                background: #eef2f7;
                color: #111827;
                font-family: "SF Pro Text", "Helvetica Neue", sans-serif;
                font-size: 13px;
            }
            QListWidget, QLineEdit {
                background: #ffffff;
                border: 1px solid #d8e0ea;
                border-radius: 12px;
                padding: 6px;
            }
            QListWidget::item {
                padding: 6px 8px;
                border-radius: 8px;
            }
            QListWidget::item:selected {
                background: #dbeafe;
                color: #0f172a;
            }
            QPushButton {
                background: #ffffff;
                border: 1px solid #d8e0ea;
                border-radius: 12px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #f8fafc;
            }
            QPushButton:pressed {
                background: #eef2f7;
            }
            """
        )

    def _reload_clips(self):
        if not self.clip_dir.exists():
            self.all_clips = []
            self.clip_list.clear()
            self.status_label.setText(f"Missing clip directory: {self.clip_dir}")
            return
        self.all_clips = sorted(self.clip_dir.glob("*.json"), key=lambda p: p.name.lower())
        self._refresh_list()
        self.status_label.setText(f"Loaded {len(self.all_clips)} clips")

    def _refresh_list(self):
        selected_path = self._selected_clip_path()
        selected_name = selected_path.name if selected_path else None
        query = self.search_input.text().strip().lower()
        self.clip_list.clear()

        for clip_path in self.all_clips:
            if query and query not in clip_path.name.lower():
                continue
            item = QListWidgetItem(clip_path.name)
            item.setData(Qt.UserRole, str(clip_path))
            self.clip_list.addItem(item)

        if self.clip_list.count() == 0:
            self._clear_details()
            return

        if selected_name:
            for i in range(self.clip_list.count()):
                item = self.clip_list.item(i)
                if item.text() == selected_name:
                    self.clip_list.setCurrentItem(item)
                    return
        self.clip_list.setCurrentRow(0)

    def _selected_clip_path(self) -> Path | None:
        item = self.clip_list.currentItem()
        if item is None:
            return None
        raw = item.data(Qt.UserRole)
        if not raw:
            return None
        return Path(str(raw))

    def _on_selected_clip_changed(self):
        clip_path = self._selected_clip_path()
        if clip_path is None:
            self._clear_details()
            return
        self._populate_details(clip_path)
        self._play_selected_clip()

    def _populate_details(self, clip_path: Path):
        stem = clip_path.stem
        inferred_token = _base_stem(stem).upper()
        self.clip_name_label.setText(f"Clip: {clip_path.name}")
        self.prints_label.setText(f"Avatar prints: {inferred_token}")

        fps = "-"
        frames = []
        parse_error = None
        try:
            with clip_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            fps_val = data.get("fps")
            frames = data.get("frames", [])
            fps = f"{float(fps_val):.2f}" if fps_val is not None else "-"
        except Exception as exc:
            parse_error = str(exc)

        frame_count = len(frames) if isinstance(frames, list) else 0
        duration = "-"
        if fps != "-" and frame_count > 0:
            try:
                duration = f"{(frame_count / float(fps)):.2f}s"
            except Exception:
                duration = "-"
        size_kb = clip_path.stat().st_size / 1024.0
        self.meta_label.setText(
            f"Frames: {frame_count} | FPS: {fps} | Duration: {duration} | Size: {size_kb:.1f} KB"
        )

        tokens = self._resolve_dictionary_tokens(stem)
        if tokens:
            self.tokens_label.setText(f"Dictionary tokens: {', '.join(tokens)}")
        else:
            self.tokens_label.setText("Dictionary tokens: (not referenced)")

        if parse_error:
            self.status_label.setText(f"JSON error: {parse_error}")
        else:
            self.status_label.setText("Ready")

    def _clip_duration(self, clip_path: Path) -> float:
        try:
            with clip_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            fps = float(data.get("fps", 0))
            frames = data.get("frames", [])
            frame_count = len(frames) if isinstance(frames, list) else 0
            if fps > 0 and frame_count > 0:
                raw_duration = frame_count / fps
                playback_speed = float(getattr(self.preview, "PLAYBACK_SPEED", 1.0) or 1.0)
                # ASLAnimationView plays clips at PLAYBACK_SPEED, so keep the event
                # alive long enough to reach the final frame.
                adjusted = raw_duration / max(playback_speed, 1e-3)
                return max(adjusted, 0.25)
        except Exception:
            pass
        return 1.0

    def _resolve_dictionary_tokens(self, stem: str) -> list[str]:
        # 1) exact clip name match from dictionary
        exact = self.clip_token_map.get(stem, [])
        if exact:
            return exact

        # 2) variant clip fallback: about_2 -> about
        base = _base_stem(stem)
        if base != stem:
            base_tokens = self.clip_token_map.get(base, [])
            if base_tokens:
                return base_tokens

        # 3) token-name fallback if filename matches token directly
        token_key = base.upper()
        if token_key in ASL_SIGNS:
            return [token_key]

        return []

    def _play_selected_clip(self):
        clip_path = self._selected_clip_path()
        if clip_path is None:
            return
        duration = self._clip_duration(clip_path)
        event = SignEvent(
            token=clip_path.stem.upper(),
            clip=clip_path.stem,
            start=0.0,
            duration=duration,
        )
        self.preview.play([event])
        self.status_label.setText(f"Playing: {clip_path.name}")

    def _select_previous(self):
        idx = self.clip_list.currentRow()
        if idx > 0:
            self.clip_list.setCurrentRow(idx - 1)

    def _select_next(self):
        idx = self.clip_list.currentRow()
        if idx < self.clip_list.count() - 1:
            self.clip_list.setCurrentRow(idx + 1)

    def _move_selected_to_bad(self):
        clip_path = self._selected_clip_path()
        if clip_path is None:
            return
        self.bad_dir.mkdir(parents=True, exist_ok=True)
        target = self.bad_dir / clip_path.name
        if target.exists():
            reply = QMessageBox.question(
                self,
                "Overwrite clip in _bad",
                f"{target.name} already exists in _bad. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
            target.unlink(missing_ok=True)
        clip_path.replace(target)
        self.status_label.setText(f"Moved to _bad: {target.name}")
        self._reload_clips()

    def _delete_selected_clip(self):
        clip_path = self._selected_clip_path()
        if clip_path is None:
            return
        reply = QMessageBox.warning(
            self,
            "Delete clip",
            f"Delete {clip_path.name} permanently?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        clip_path.unlink(missing_ok=True)
        self.status_label.setText(f"Deleted: {clip_path.name}")
        self._reload_clips()

    def _clear_details(self):
        self.clip_name_label.setText("Clip: (none)")
        self.prints_label.setText("Avatar prints: (none)")
        self.meta_label.setText("Frames: - | FPS: - | Duration: - | Size: -")
        self.tokens_label.setText("Dictionary tokens: (none)")
        self.status_label.setText("No clips found")
