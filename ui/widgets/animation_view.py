import time
from typing import List
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QPen
from core.sequencing.sign_sequencer import SignEvent
from ui.animation.poses.basic_poses import POSES
from ui.animation.clip_loader import load_clip


class ASLAnimationView(QWidget):
    def __init__(self):
        super().__init__()

        self.sequence: List[SignEvent] = []
        self.start_time = None

        self.live_pose = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.clip = None
        self.clip_start_time = None
        self.clip_frame_index = 0

    @staticmethod
    def interpolate_pose(pose_a, pose_b, t):
        result = {}
        for joint in pose_a:
            x0, y0 = pose_a[joint]
            x1, y1 = pose_b.get(joint, (x0, y0))
            result[joint] = (
                x0 + (x1 - x0) * t,
                y0 + (y1 - y0) * t
            )
        return result

    def play(self, sequence: List[SignEvent]):
        self.sequence = sequence
        self.start_time = time.time()

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(Qt.black, 4)
            painter.setPen(pen)

            w, h = self.width(), self.height()
            pose = self.live_pose if self.live_pose else self._get_active_pose()
            def p(name):
                if name not in pose:
                    return None
                x, y = pose[name]
                return int(x * w), int(y * h)

            self._line(painter, p("head"), p("torso"))

            self._line(painter, p("shoulder_left"), p("elbow_left"))
            self._line(painter, p("elbow_left"), p("hand_left"))

            self._line(painter, p("shoulder_right"), p("elbow_right"))
            self._line(painter, p("elbow_right"), p("hand_right"))
        finally:
            painter.end()

    def _get_active_pose(self):
        if self.live_pose:
            return self.live_pose

        if not self.sequence or self.start_time is None:
            return POSES["REST"]

        elapsed = time.time() - self.start_time

        for e in self.sequence:
            if e.start <= elapsed < e.start + e.duration:

                # Load clip once per sign
                if self.clip is None or self.clip["sign"] != e.sign:
                    clip_data = load_clip(e.sign)
                    if clip_data is None:
                        return POSES.get(e.sign, POSES["REST"])

                    self.clip = {
                        "sign": e.sign,
                        "frames": clip_data["frames"],
                        "fps": clip_data["fps"]
                    }
                    self.clip_start_time = time.time()
                    self.clip_frame_index = 0

                # Advance frames
                frame_time = 1.0 / self.clip["fps"]
                frame_count = len(self.clip["frames"])
                elapsed_clip = time.time() - self.clip_start_time

                index = int(elapsed_clip / frame_time)
                index = min(index, frame_count - 1)

                return self.clip["frames"][index]

        self.clip = None
        return POSES["REST"]

    def _line(self, painter, a, b):
        if a is None or b is None:
            return
        painter.drawLine(a[0], a[1], b[0], b[1])

    def set_live_pose(self, pose: dict):
        self.live_pose = pose
        self.update()
        print("LIVE POSE KEYS:", pose.keys())

    def enable_live_mode(self, enabled: bool):
        if enabled:
            self.sequence = []
            self.clip = None
            self.start_time = None