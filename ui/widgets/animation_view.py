import time
from typing import List
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QLinearGradient, QRadialGradient
from core.sequencing.sign_sequencer import SignEvent
from ui.animation.poses.basic_poses import POSES
from ui.animation.clip_loader import load_clip


class ASLAnimationView(QWidget):
    HAND_VISUAL_SCALE = 0.84
    # Keep motion smooth but still readable for live demo.
    PLAYBACK_SPEED = 0.62
    PLAYBACK_SMOOTHING_ALPHA = 0.20
    HAND_CHAINS = (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    FINGER_COLORS = (
        QColor(116, 226, 185),
        QColor(113, 206, 248),
        QColor(124, 176, 255),
        QColor(156, 155, 250),
        QColor(233, 156, 206),
    )
    BODY_COLOR = QColor(246, 84, 92)
    BODY_SHADOW = QColor(18, 22, 30, 190)
    JOINT_COLOR = QColor(244, 247, 255)
    VIEWPORT_LEFT = 0.08
    VIEWPORT_RIGHT = 0.92
    VIEWPORT_TOP = 0.04
    VIEWPORT_BOTTOM = 0.90
    FIT_TOP_PADDING = 0.08
    FIT_BOTTOM_PADDING = 0.24

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
            painter.setClipRect(self.rect())

            w, h = self.width(), self.height()
            bg = QLinearGradient(0, 0, 0, h)
            bg.setColorAt(0.0, QColor(10, 15, 22, 210))
            bg.setColorAt(1.0, QColor(4, 8, 14, 210))
            painter.fillRect(self.rect(), bg)

            vignette = QRadialGradient(w * 0.5, h * 0.55, max(w, h) * 0.62)
            vignette.setColorAt(0.0, QColor(0, 0, 0, 0))
            vignette.setColorAt(1.0, QColor(0, 0, 0, 120))
            painter.fillRect(self.rect(), vignette)
            pose = (
                self.live_pose
                if self.use_live_pose and self.live_pose
                else self._get_active_pose()
            )
            p = self._build_pose_mapper(pose, w, h)

            painter.setPen(QPen(self.BODY_SHADOW, 10, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            head = p("head")
            if head:
                self._draw_face(painter, (head[0] + 2, head[1] + 2), w, h)

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

            shadow_off = (2, 2)
            painter.setPen(QPen(self.BODY_SHADOW, 9, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            torso_shoulder_l_s, torso_shoulder_r_s = self._draw_torso(
                painter, shoulder_l, shoulder_r, h, offset=shadow_off
            )
            arm_start_l_s = torso_shoulder_l_s if torso_shoulder_l_s is not None else (
                (shoulder_l[0] + shadow_off[0], shoulder_l[1] + shadow_off[1]) if shoulder_l else None
            )
            arm_start_r_s = torso_shoulder_r_s if torso_shoulder_r_s is not None else (
                (shoulder_r[0] + shadow_off[0], shoulder_r[1] + shadow_off[1]) if shoulder_r else None
            )
            elbow_l_s = (elbow_l[0] + shadow_off[0], elbow_l[1] + shadow_off[1]) if elbow_l else None
            elbow_r_s = (elbow_r[0] + shadow_off[0], elbow_r[1] + shadow_off[1]) if elbow_r else None
            wrist_l_s = (wrist_l[0] + shadow_off[0], wrist_l[1] + shadow_off[1]) if wrist_l else None
            wrist_r_s = (wrist_r[0] + shadow_off[0], wrist_r[1] + shadow_off[1]) if wrist_r else None
            self._line(painter, arm_start_l_s, elbow_l_s)
            self._line(painter, elbow_l_s, wrist_l_s)
            self._line(painter, arm_start_r_s, elbow_r_s)
            self._line(painter, elbow_r_s, wrist_r_s)
            self._draw_full_hand(painter, p, "left", offset=shadow_off, shadow=True)
            self._draw_full_hand(painter, p, "right", offset=shadow_off, shadow=True)

            painter.setPen(QPen(self.BODY_COLOR, 6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
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

    def _build_pose_mapper(self, pose: dict, width: int, height: int):
        points = []
        for value in pose.values():
            try:
                x, y = value
                points.append((float(x), float(y)))
            except Exception:
                continue

        if not points:
            def _empty_mapper(name):
                return None
            return _empty_mapper

        min_x = min(x for x, _ in points)
        max_x = max(x for x, _ in points)
        min_y = min(y for _, y in points)
        max_y = max(y for _, y in points)

        span_x = max(max_x - min_x, 1e-3)
        span_y = max(max_y - min_y, 1e-3)

        left = self.VIEWPORT_LEFT * width
        right = self.VIEWPORT_RIGHT * width
        top = self.VIEWPORT_TOP * height
        bottom = self.VIEWPORT_BOTTOM * height
        view_w = max(1.0, right - left)
        view_h = max(1.0, bottom - top)
        top_pad_px = self.FIT_TOP_PADDING * height
        bottom_pad_px = self.FIT_BOTTOM_PADDING * height
        fit_h = max(1.0, view_h - top_pad_px - bottom_pad_px)

        scale = min(view_w / span_x, fit_h / span_y)
        used_w = span_x * scale
        used_h = span_y * scale
        offset_x = left + (view_w - used_w) * 0.5
        offset_y = top + top_pad_px + (fit_h - used_h) * 0.5

        def _mapper(name):
            if name not in pose:
                return None
            try:
                x, y = pose[name]
            except Exception:
                return None
            px = offset_x + (float(x) - min_x) * scale
            py = offset_y + (float(y) - min_y) * scale
            return int(px), int(py)

        return _mapper

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

    def _draw_full_hand(self, painter, p, side: str, offset=(0, 0), shadow=False):
        prefix = f"{side}_hand_"
        wrist_base = p("hand_left" if side == "left" else "hand_right")
        if wrist_base is None:
            return
        wrist = (wrist_base[0] + offset[0], wrist_base[1] + offset[1])

        def hp(idx):
            pt = p(f"{prefix}{idx}")
            if pt is None:
                return None
            sx = wrist_base[0] + (pt[0] - wrist_base[0]) * self.HAND_VISUAL_SCALE + offset[0]
            sy = wrist_base[1] + (pt[1] - wrist_base[1]) * self.HAND_VISUAL_SCALE + offset[1]
            return int(sx), int(sy)

        hand0 = hp(0)
        if hand0:
            color = self.BODY_SHADOW if shadow else self.BODY_COLOR
            width = 8 if shadow else 6
            painter.setPen(QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            self._line(painter, wrist, hand0)

        for i, chain in enumerate(self.HAND_CHAINS):
            finger_color = self.BODY_SHADOW if shadow else self.FINGER_COLORS[i]
            finger_width = 7 if shadow else 5
            painter.setPen(QPen(finger_color, finger_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            for j in range(len(chain) - 1):
                self._line(painter, hp(chain[j]), hp(chain[j + 1]))

        joint_color = QColor(self.BODY_SHADOW) if shadow else self.JOINT_COLOR
        painter.setPen(QPen(joint_color, 1))
        painter.setBrush(joint_color)
        for idx in range(21):
            self._dot(painter, hp(idx), radius=5 if shadow else 4)

    def _draw_torso(self, painter, shoulder_l, shoulder_r, h, offset=(0, 0)):
        if shoulder_l is None or shoulder_r is None:
            return None, None
        cx = (shoulder_l[0] + shoulder_r[0]) // 2
        shoulder_span = shoulder_r[0] - shoulder_l[0]
        torso_span = int(shoulder_span * 0.62)
        top_y = int((shoulder_l[1] + shoulder_r[1]) / 2) + 4
        # Extend torso close to the lower viewport so full body reads naturally.
        target_bottom_y = int(h * 0.94)
        body_height = max(int(shoulder_span * 0.95), target_bottom_y - top_y)
        top_l = (cx - torso_span // 2 + offset[0], top_y + offset[1])
        top_r = (cx + torso_span // 2 + offset[0], top_y + offset[1])
        hip_l = (cx - int(torso_span * 0.46) + offset[0], top_y + body_height + offset[1])
        hip_r = (cx + int(torso_span * 0.44) + offset[0], top_y + body_height + 5 + offset[1])
        self._line(painter, top_l, top_r)
        self._line(painter, top_l, hip_l)
        self._line(painter, top_r, hip_r)
        self._line(painter, hip_l, hip_r)
        return top_l, top_r

    def _draw_face(self, painter, head, w, h):
        rw = max(30, int(w * 0.034))
        rh = max(44, int(h * 0.086))
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
        painter.drawArc(cx - int(rw * 0.46), mouth_y - 8, int(rw * 0.92), 20, 200 * 16, 140 * 16)

    def _naturalize_shoulders(self, head, shoulder_l, shoulder_r, h):
        if head is None or shoulder_l is None or shoulder_r is None:
            return shoulder_l, shoulder_r

        # Pull shoulder line upward toward a natural head-neck distance.
        target_y = head[1] + int(h * 0.12)
        ly = int(shoulder_l[1] * 0.45 + target_y * 0.55)
        ry = int(shoulder_r[1] * 0.45 + target_y * 0.55)
        return (shoulder_l[0], ly), (shoulder_r[0], ry)
