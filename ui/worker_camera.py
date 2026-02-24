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
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def run(self):
        if cv2 is None or mp is None:
            self.error_ready.emit("Camera dependencies unavailable (cv2/mediapipe)")
            print("Camera dependencies unavailable (cv2/mediapipe)")
            return
        from core.vision.pose_adapter import mediapipe_to_pose_dict

        camera_index = int(os.getenv("ASL_CAMERA_INDEX", "0"))
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

        # Prefer higher quality preview for the UI feed.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        mp_holistic = mp.solutions.holistic.Holistic(
            model_complexity=0,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        thread = QThread.currentThread()
        failed_reads = 0

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

            # Run landmark detection on a smaller frame for stable throughput.
            proc = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
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
