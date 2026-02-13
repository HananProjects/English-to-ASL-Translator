import time
from typing import List
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from core.sequencing.sign_sequencer import SignEvent
from ui.animation.poses.basic_poses import POSES
from ui.animation.clip_loader import load_clip


class ASLAnimationView(QWidget):
    HAND_VISUAL_SCALE = 0.78
    PLAYBACK_SPEED = 0.75
    PLAYBACK_SMOOTHING_ALPHA = 0.35
    HAND_CHAINS = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    FINGER_COLORS = (
        QColor(120, 255, 140),
        QColor(80, 230, 255),
        QColor(90, 170, 255),
        QColor(170, 130, 255),
        QColor(255, 130, 220),
    )
    BODY_COLOR = QColor(210, 10, 20)
    JOINT_COLOR = QColor(245, 245, 245)

    def __init__(self):
        super().__init__()
        self.setAutoFillBackground(False)

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
                return (
                    int((0.04 + x * 0.92) * w),
                    int((0.02 + y * 0.94) * h),
                )

            painter.setPen(QPen(self.BODY_COLOR, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            head = p("head")
            if head:
                self._draw_face(painter, head, w, h)

            shoulder_l = p("shoulder_left")
            shoulder_r = p("shoulder_right")
            elbow_l = p("elbow_left")
            elbow_r = p("elbow_right")
            wrist_l = p("hand_left")
            wrist_r = p("hand_right")
            shoulder_l, shoulder_r = self._naturalize_shoulders(
                head,
                shoulder_l,
                shoulder_r,
                h,
            )

            torso_shoulder_l, torso_shoulder_r = self._draw_torso(
                painter, shoulder_l, shoulder_r, h
            )
            arm_start_l = torso_shoulder_l if torso_shoulder_l is not None else shoulder_l
            arm_start_r = torso_shoulder_r if torso_shoulder_r is not None else shoulder_r

            self._line(painter, arm_start_l, elbow_l)
            self._line(painter, elbow_l, wrist_l)
            self._line(painter, arm_start_r, elbow_r)
            self._line(painter, elbow_r, wrist_r)

            self._draw_full_hand(painter, p, "left")
            self._draw_full_hand(painter, p, "right")

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
        if hand0:
            painter.setPen(QPen(self.BODY_COLOR, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            self._line(painter, wrist, hand0)

        for i, chain in enumerate(self.HAND_CHAINS):
            painter.setPen(QPen(self.FINGER_COLORS[i], 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            for j in range(len(chain) - 1):
                self._line(painter, hp(chain[j]), hp(chain[j + 1]))

        painter.setPen(QPen(self.JOINT_COLOR, 1))
        painter.setBrush(self.JOINT_COLOR)
        for idx in range(21):
            self._dot(painter, hp(idx), radius=4)

    def _draw_torso(self, painter, shoulder_l, shoulder_r, h):
        if shoulder_l is None or shoulder_r is None:
            return None, None
        cx = (shoulder_l[0] + shoulder_r[0]) // 2
        shoulder_span = shoulder_r[0] - shoulder_l[0]
        torso_span = int(shoulder_span * 0.55)
        top_y = int((shoulder_l[1] + shoulder_r[1]) / 2) + 4
        body_height = int(h * 0.22)
        top_l = (cx - torso_span // 2, top_y)
        top_r = (cx + torso_span // 2, top_y)
        hip_l = (cx - int(torso_span * 0.42), top_y + body_height)
        hip_r = (cx + int(torso_span * 0.38), top_y + body_height + 6)
        self._line(painter, top_l, top_r)
        self._line(painter, top_l, hip_l)
        self._line(painter, top_r, hip_r)
        self._line(painter, hip_l, hip_r)
        return top_l, top_r

    def _draw_face(self, painter, head, w, h):
        rw = max(26, int(w * 0.03))
        rh = max(40, int(h * 0.08))
        cx, cy = head
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(cx - rw, cy - rh, rw * 2, rh * 2)

        eye_y = cy - int(rh * 0.15)
        brow_y = cy - int(rh * 0.38)
        eye_dx = int(rw * 0.45)
        eye_w = int(rw * 0.35)
        mouth_y = cy + int(rh * 0.42)
        painter.drawArc(cx - eye_dx - eye_w, eye_y - 5, eye_w * 2, 10, 0, 180 * 16)
        painter.drawArc(cx + eye_dx - eye_w, eye_y - 5, eye_w * 2, 10, 0, 180 * 16)
        painter.drawLine(cx - eye_dx - eye_w, brow_y, cx - eye_dx + eye_w, brow_y - 4)
        painter.drawLine(cx + eye_dx - eye_w, brow_y - 4, cx + eye_dx + eye_w, brow_y)
        painter.drawArc(cx - int(rw * 0.45), mouth_y - 8, int(rw * 0.9), 20, 200 * 16, 140 * 16)

    def _naturalize_shoulders(self, head, shoulder_l, shoulder_r, h):
        if head is None or shoulder_l is None or shoulder_r is None:
            return shoulder_l, shoulder_r

        # Pull shoulder line upward toward a natural head-neck distance.
        target_y = head[1] + int(h * 0.12)
        ly = int(shoulder_l[1] * 0.45 + target_y * 0.55)
        ry = int(shoulder_r[1] * 0.45 + target_y * 0.55)
        return (shoulder_l[0], ly), (shoulder_r[0], ry)
