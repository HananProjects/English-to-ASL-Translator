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
        self.current_pose = POSES["REST"]

        self.prev_pose = POSES["REST"]
        self.target_pose = POSES["REST"]
        self.pose_start_time = None
        self.pose_duration = 0.3  # seconds (tweakable)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

        self.clip = None
        self.clip_start_time = None
        self.clip_frame_index = 0

    def interpolate_pose(self, pose_a, pose_b, t):        
        """
        Linearly interpolate between two poses.
        t in [0,1]
        """
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

            pose = self._get_active_pose()
            if not pose:
                return

            w, h = self.width(), self.height()

            def p(name):
                x, y = pose[name]
                return int(x * w), int(y * h)

            self._line(painter, p("head"), p("shoulder"))
            self._line(painter, p("shoulder"), p("elbow_left"))
            self._line(painter, p("elbow_left"), p("hand_left"))
            self._line(painter, p("shoulder"), p("elbow_right"))
            self._line(painter, p("elbow_right"), p("hand_right"))
            self._line(painter, p("shoulder"), p("torso"))
        finally:
            painter.end()

    def _get_active_pose(self):
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
        painter.drawLine(a[0], a[1], b[0], b[1])