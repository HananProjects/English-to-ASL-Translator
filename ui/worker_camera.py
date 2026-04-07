import os

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None
from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtGui import QImage
from core.asl_to_english.recognizer import SignStreamRecognizer
from core.vision.live_pose import LivePoseFilter


def _env_int(name: str, default: int, min_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    if min_value is not None and value < min_value:
        return min_value
    return value


def _env_float(name: str, default: float, min_value: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    if min_value is not None and value < min_value:
        return min_value
    return value


def _camera_probe_indices(start_index: int, probe_count: int) -> list[int]:
    ordered = [start_index]
    for index in range(probe_count):
        if index not in ordered:
            ordered.append(index)
    return ordered


def _open_camera_capture(start_index: int, probe_count: int):
    attempted: list[int] = []
    for index in _camera_probe_indices(start_index, probe_count):
        attempted.append(index)
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap, index, attempted
        cap.release()
    return None, None, attempted


class CameraWorker(QObject):
    pose_ready = Signal(dict)
    frame_ready = Signal(object)
    token_ready = Signal(str, float)
    sequence_ready = Signal(list)
    debug_ready = Signal(str, float, int)
    error_ready = Signal(str)
    finished = Signal(bool)

    def __init__(self, source_path: str | None = None, fast_mode: bool = False):
        super().__init__()
        self.fast_mode = bool(fast_mode)
        # Fast mode trades some strictness for responsiveness on constrained hardware.
        stable_frames = _env_int("ASL_STABLE_FRAMES", 4 if self.fast_mode else 3, min_value=1)
        min_confidence = _env_float(
            "ASL_MIN_CONFIDENCE",
            0.58 if self.fast_mode else 0.52,
            min_value=0.0,
        )
        emit_cooldown_frames = _env_int(
            "ASL_EMIT_COOLDOWN_FRAMES",
            6 if self.fast_mode else 4,
            min_value=0,
        )
        pause_frames = _env_int(
            "ASL_PAUSE_FRAMES",
            10 if self.fast_mode else 10,
            min_value=1,
        )
        # Prefer model/hybrid matching; recognizer will use v2 model if available.
        self.recognizer = SignStreamRecognizer(
            prefer_model=True,
            stable_frames=stable_frames,
            min_confidence=min_confidence,
            emit_cooldown_frames=emit_cooldown_frames,
            pause_frames=pause_frames,
        )
        self.source_path = source_path
        self._running = False
        self._cap = None
        smoothing_alpha = _env_float("ASL_POSE_SMOOTHING", 0.50, min_value=0.0)
        hand_hold_frames = _env_int("ASL_HAND_HOLD_FRAMES", 3, min_value=0)
        body_hold_frames = _env_int("ASL_BODY_HOLD_FRAMES", 1, min_value=0)
        self.pose_filter = LivePoseFilter(
            alpha=min(1.0, smoothing_alpha),
            hand_hold_frames=hand_hold_frames,
            body_hold_frames=body_hold_frames,
        )
        print(
            "[ASL] recognizer matcher="
            f"{type(self.recognizer.matcher).__name__} "
            f"mode={'FAST' if self.fast_mode else 'NORMAL'} "
            f"smoothing={self.pose_filter.alpha:.2f} "
            f"hand_hold={hand_hold_frames} body_hold={body_hold_frames}"
        )

    def reset_recognition_state(self):
        self.recognizer.reset()
        self.pose_filter.reset()

    def stop(self):
        self._running = False
        # Releasing capture can unblock a pending read() during camera faults.
        cap = self._cap
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass

    def run(self):
        if cv2 is None or mp is None:
            self.error_ready.emit("Camera dependencies unavailable (cv2/mediapipe)")
            print("Camera dependencies unavailable (cv2/mediapipe)")
            return
        from core.vision.pose_adapter import mediapipe_to_pose_dict

        default_width = 640 if self.fast_mode else 1280
        default_height = 480 if self.fast_mode else 720
        default_proc_width = 384 if self.fast_mode else 640
        default_proc_height = 216 if self.fast_mode else 360
        fast_try_mjpg = os.getenv("ASL_CAMERA_FAST_MJPG", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        camera_index = _env_int("ASL_CAMERA_INDEX", 0, min_value=0)
        camera_width = _env_int("ASL_CAMERA_WIDTH", default_width, min_value=160)
        camera_height = _env_int("ASL_CAMERA_HEIGHT", default_height, min_value=120)
        camera_fps = _env_int("ASL_CAMERA_FPS", 30, min_value=1)
        camera_buffer_size = _env_int("ASL_CAMERA_BUFFER_SIZE", 1, min_value=1)
        camera_probe_count = _env_int("ASL_CAMERA_PROBE_COUNT", 4, min_value=1)
        proc_width = _env_int("ASL_PROCESS_WIDTH", default_proc_width, min_value=160)
        proc_height = _env_int("ASL_PROCESS_HEIGHT", default_proc_height, min_value=120)
        process_every_n = _env_int("ASL_PROCESS_EVERY_N", 1, min_value=1)
        zoom = _env_float("ASL_CAMERA_ZOOM", 1.0, min_value=1.0)
        source_is_file = bool(self.source_path)

        selected_camera_index = camera_index
        if source_is_file:
            cap = cv2.VideoCapture(self.source_path)
            attempted_indices: list[int] = []
        else:
            cap, detected_index, attempted_indices = _open_camera_capture(
                camera_index,
                camera_probe_count,
            )
            if detected_index is not None:
                selected_camera_index = detected_index
        self._cap = cap
        self._running = True
        if cap is None or not cap.isOpened():
            if source_is_file:
                self.error_ready.emit(
                    f"Unable to open video file: {self.source_path}"
                )
            else:
                attempts_text = ", ".join(str(index) for index in attempted_indices)
                self.error_ready.emit(
                    f"Unable to open camera. Tried indices: {attempts_text}. "
                    "Set ASL_CAMERA_INDEX or increase ASL_CAMERA_PROBE_COUNT."
                )
            self._running = False
            if cap is not None:
                cap.release()
            self._cap = None
            return

        if not source_is_file:
            if self.fast_mode and fast_try_mjpg:
                try:
                    cap.set(
                        cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*"MJPG"),
                    )
                except Exception:
                    pass
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
            cap.set(cv2.CAP_PROP_FPS, camera_fps)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, camera_buffer_size)
            except Exception:
                pass
            print(
                "[Camera] index="
                f"{selected_camera_index} {camera_width}x{camera_height}@{camera_fps} "
                f"proc={proc_width}x{proc_height} skip={process_every_n} zoom={zoom:.2f} "
                f"buffer={camera_buffer_size} mode={'FAST' if self.fast_mode else 'NORMAL'}"
            )
        else:
            print(
                "[Video] file="
                f"{self.source_path} proc={proc_width}x{proc_height} "
                f"skip={process_every_n}"
            )

        mp_holistic = mp.solutions.holistic.Holistic(
            model_complexity=0,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.40,
            min_tracking_confidence=0.40
        )

        thread = QThread.currentThread()
        failed_reads = 0
        frame_index = 0

        while self._running and cap.isOpened() and not thread.isInterruptionRequested():
            ret, frame = cap.read()
            if not ret:
                if source_is_file:
                    break
                failed_reads += 1
                if failed_reads >= 600:
                    self.error_ready.emit("Camera stream read failed repeatedly.")
                    break
                QThread.msleep(5)
                continue
            failed_reads = 0
            frame_index += 1

            if zoom > 1.0:
                h0, w0 = frame.shape[:2]
                crop_w = max(2, int(w0 / zoom))
                crop_h = max(2, int(h0 / zoom))
                x0 = (w0 - crop_w) // 2
                y0 = (h0 - crop_h) // 2
                frame = frame[y0:y0 + crop_h, x0:x0 + crop_w]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            bytes_per_line = c * w
            frame_image = QImage(
                rgb.data,
                w,
                h,
                bytes_per_line,
                QImage.Format_RGB888
            ).copy()
            self.frame_ready.emit(frame_image)

            if process_every_n > 1 and (frame_index % process_every_n) != 0:
                continue

            # Run landmark detection on a smaller frame for stable throughput.
            proc = cv2.resize(rgb, (proc_width, proc_height), interpolation=cv2.INTER_AREA)
            result = mp_holistic.process(proc)

            # Extract all landmark groups safely
            pose_landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
            left_hand_landmarks = result.left_hand_landmarks.landmark if result.left_hand_landmarks else None
            right_hand_landmarks = result.right_hand_landmarks.landmark if result.right_hand_landmarks else None

            if not pose_landmarks:
                self.debug_ready.emit("", 0.0, 0)
                continue

            pose = mediapipe_to_pose_dict(
                pose_landmarks,
                left_hand_landmarks,
                right_hand_landmarks
            )
            pose = self.pose_filter.apply(pose)
            self.pose_ready.emit(pose)
            update = self.recognizer.process(pose)
            raw_token = update.raw_token or ""
            self.debug_ready.emit(
                raw_token,
                float(update.raw_confidence),
                int(update.candidate_streak),
            )
            if update.detected_token is not None:
                self.token_ready.emit(update.detected_token, update.confidence)
            if update.sentence_tokens is not None:
                self.sequence_ready.emit(update.sentence_tokens)

        if source_is_file and self.recognizer.buffered_tokens:
            self.sequence_ready.emit(list(self.recognizer.buffered_tokens))
            self.recognizer.buffered_tokens.clear()
        self._running = False
        try:
            mp_holistic.close()
        except Exception:
            pass
        try:
            cap.release()
        except Exception:
            pass
        self._cap = None
        self.finished.emit(source_is_file)
        print("Camera thread exiting cleanly")
