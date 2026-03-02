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


class CameraWorker(QObject):
    pose_ready = Signal(dict)
    frame_ready = Signal(object)
    token_ready = Signal(str, float)
    sequence_ready = Signal(list)
    debug_ready = Signal(str, float, int)
    error_ready = Signal(str)

    def __init__(self):
        super().__init__()
        # Prefer model/hybrid matching; recognizer will use v2 model if available.
        self.recognizer = SignStreamRecognizer(
            prefer_model=True,
            stable_frames=6,
            min_confidence=0.60,
            emit_cooldown_frames=10,
            pause_frames=18,
        )
        self._running = False
        self._cap = None
        print(f"[ASL] recognizer matcher={type(self.recognizer.matcher).__name__}")

    def reset_recognition_state(self):
        self.recognizer.reset()

    def stop(self):
        self._running = False

    def run(self):
        if cv2 is None or mp is None:
            self.error_ready.emit("Camera dependencies unavailable (cv2/mediapipe)")
            print("Camera dependencies unavailable (cv2/mediapipe)")
            return
        from core.vision.pose_adapter import mediapipe_to_pose_dict

        camera_index = _env_int("ASL_CAMERA_INDEX", 0, min_value=0)
        camera_width = _env_int("ASL_CAMERA_WIDTH", 1280, min_value=160)
        camera_height = _env_int("ASL_CAMERA_HEIGHT", 720, min_value=120)
        camera_fps = _env_int("ASL_CAMERA_FPS", 30, min_value=1)
        camera_buffer_size = _env_int("ASL_CAMERA_BUFFER_SIZE", 1, min_value=1)
        proc_width = _env_int("ASL_PROCESS_WIDTH", 640, min_value=160)
        proc_height = _env_int("ASL_PROCESS_HEIGHT", 360, min_value=120)
        process_every_n = _env_int("ASL_PROCESS_EVERY_N", 1, min_value=1)
        zoom = _env_float("ASL_CAMERA_ZOOM", 1.0, min_value=1.0)

        cap = cv2.VideoCapture(camera_index)
        self._cap = cap
        self._running = True
        if not cap.isOpened():
            self.error_ready.emit(
                f"Unable to open camera index {camera_index}. "
                "Set ASL_CAMERA_INDEX to the correct camera."
            )
            self._running = False
            cap.release()
            self._cap = None
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)
        cap.set(cv2.CAP_PROP_FPS, camera_fps)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, camera_buffer_size)
        except Exception:
            pass
        print(
            "[Camera] index="
            f"{camera_index} {camera_width}x{camera_height}@{camera_fps} "
            f"proc={proc_width}x{proc_height} skip={process_every_n} zoom={zoom:.2f} "
            f"buffer={camera_buffer_size}"
        )

        mp_holistic = mp.solutions.holistic.Holistic(
            model_complexity=0,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        thread = QThread.currentThread()
        failed_reads = 0
        frame_index = 0

        while self._running and cap.isOpened() and not thread.isInterruptionRequested():
            ret, frame = cap.read()
            if not ret:
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

        self._running = False
        try:
            mp_holistic.close()
        except Exception:
            pass
        cap.release()
        self._cap = None
        print("Camera thread exiting cleanly")
