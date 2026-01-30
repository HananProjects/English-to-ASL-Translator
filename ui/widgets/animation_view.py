import time
from typing import List
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QPen
from core.sequencing.sign_sequencer import SignEvent
from ui.animation.poses.basic_poses import POSES


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
                target = POSES.get(e.sign, POSES["REST"])

                if self.target_pose != target:
                    self.prev_pose = self.current_pose
                    self.target_pose = target
                    self.pose_start_time = time.time()

                # interpolation factor
                t = min(
                    (time.time() - self.pose_start_time) / self.pose_duration,
                    1.0
                )

                self.current_pose = self.interpolate_pose(
                    self.prev_pose,
                    self.target_pose,
                    t
                )

                return self.current_pose

        return POSES["REST"]

    def _line(self, painter, a, b):
        painter.drawLine(a[0], a[1], b[0], b[1])