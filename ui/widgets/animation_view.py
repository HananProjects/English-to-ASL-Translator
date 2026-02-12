import time
from typing import List
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QPen
from core.sequencing.sign_sequencer import SignEvent
from ui.animation.poses.basic_poses import POSES
from ui.animation.clip_loader import load_clip


class ASLAnimationView(QWidget):
    HAND_VISUAL_SCALE = 0.75
    PLAYBACK_SPEED = 0.75
    PLAYBACK_SMOOTHING_ALPHA = 0.35
    HAND_CONNECTIONS = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )

    def __init__(self):
        super().__init__()

        self.sequence: List[SignEvent] = []
        self.start_time = None

        self.live_pose = None
        self.use_live_pose = True

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(16)

        self.clip = None
        self.clip_start_time = None
        self.clip_frame_index = 0
        self.smoothed_pose = None

    @staticmethod
    def interpolate_pose(pose_a, pose_b, t):
        result = {}
        joints = set(pose_a.keys()) | set(pose_b.keys())
        for joint in joints:
            x0, y0 = pose_a.get(joint, pose_b.get(joint))
            x1, y1 = pose_b.get(joint, (x0, y0))
            result[joint] = (
                x0 + (x1 - x0) * t,
                y0 + (y1 - y0) * t
            )
        return result

    def play(self, sequence: List[SignEvent]):
        self.sequence = sequence
        self.start_time = time.time()
        self.smoothed_pose = None

    def paintEvent(self, event):
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(Qt.black, 4)
            painter.setPen(pen)

            w, h = self.width(), self.height()
            pose = (
                self.live_pose
                if self.use_live_pose and self.live_pose
                else self._get_active_pose()
            )   

            def p(name):
                if name not in pose:
                    return None
                x, y = pose[name]
                x = max(0.0, min(1.0, float(x)))
                y = max(0.0, min(1.0, float(y)))
                return int(x * w), int(y * h)
            
            head = p("head")
            if head:
                painter.setPen(QPen(Qt.black, 3))
                painter.setBrush(Qt.white)
                painter.drawEllipse(
                    head[0] - 12,
                    head[1] - 12,
                    24,
                    24
                )

            self._line(painter, p("shoulder_left"), p("elbow_left"))
            self._line(painter, p("elbow_left"), p("hand_left"))
            self._line(painter, p("shoulder_right"), p("elbow_right"))
            self._line(painter, p("elbow_right"), p("hand_right"))

            self._draw_full_hand(painter, p, "left")
            self._draw_full_hand(painter, p, "right")

            self._dot(painter, p("left_thumb_tip"))
            self._dot(painter, p("left_index_tip"))
            self._dot(painter, p("right_thumb_tip"))
            self._dot(painter, p("right_index_tip"))

        finally:
            painter.end()

    def _get_active_pose(self):
        if not self.sequence or self.start_time is None:
            return POSES["REST"]

        elapsed = time.time() - self.start_time

        for e in self.sequence:
            if e.start <= elapsed < e.start + e.duration:

                # Load clip once per sign
                if self.clip is None or self.clip["clip"] != e.clip:
                    clip_data = load_clip(e.clip)

                    if clip_data is None:
                        return POSES["REST"]

                    self.clip = {
                        "clip": e.clip,
                        "frames": clip_data["frames"],
                        "fps": clip_data["fps"]
                    }

                    self.clip_start_time = time.time()
                    self.clip_frame_index = 0

                # Advance frames
                frame_count = len(self.clip["frames"])
                elapsed_clip = time.time() - self.clip_start_time

                frame_pos = elapsed_clip * self.clip["fps"] * self.PLAYBACK_SPEED
                base_index = int(frame_pos)
                base_index = min(base_index, frame_count - 1)
                next_index = min(base_index + 1, frame_count - 1)
                frac = max(0.0, min(1.0, frame_pos - base_index))

                base_pose = self.clip["frames"][base_index]
                next_pose = self.clip["frames"][next_index]
                target_pose = self.interpolate_pose(base_pose, next_pose, frac)
                return self._smooth_playback_pose(target_pose)

        self.clip = None
        self.smoothed_pose = None
        return POSES["REST"]

    def _line(self, painter, a, b):
        if a is None or b is None:
            return
        painter.drawLine(a[0], a[1], b[0], b[1])


    def set_live_pose(self, pose: dict):
        if not self.use_live_pose:
            return  # ignore camera during ASL playback

        self.live_pose = pose
        self.update()

    def disable_live_pose(self):
        self.use_live_pose = False
        self.live_pose = None   

    def enable_live_pose(self):
        self.use_live_pose = True

    def _smooth_playback_pose(self, pose: dict):
        if self.smoothed_pose is None:
            self.smoothed_pose = pose
            return pose
        self.smoothed_pose = self.interpolate_pose(
            self.smoothed_pose,
            pose,
            self.PLAYBACK_SMOOTHING_ALPHA
        )
        return self.smoothed_pose

    def _dot(self, painter, pt, radius=3):
        if pt is None:
            return
        painter.drawEllipse(pt[0] - radius, pt[1] - radius, radius * 2, radius * 2)

    def _draw_full_hand(self, painter, p, side: str):
        prefix = f"{side}_hand_"
        wrist = p("hand_left" if side == "left" else "hand_right")
        if wrist is None:
            return

        def hp(idx):
            pt = p(f"{prefix}{idx}")
            if pt is None:
                return None
            sx = wrist[0] + (pt[0] - wrist[0]) * self.HAND_VISUAL_SCALE
            sy = wrist[1] + (pt[1] - wrist[1]) * self.HAND_VISUAL_SCALE
            return int(sx), int(sy)

        hand0 = hp(0)
        self._line(painter, wrist, hand0)

        for a, b in self.HAND_CONNECTIONS:
            self._line(painter, hp(a), hp(b))

        for idx in range(21):
            self._dot(painter, hp(idx), radius=2)
