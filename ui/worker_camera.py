try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None
from PySide6.QtCore import QObject, Signal, QThread
from core.asl_to_english.recognizer import SignStreamRecognizer


class CameraWorker(QObject):
    pose_ready = Signal(dict)
    token_ready = Signal(str, float)
    sequence_ready = Signal(list)

    def __init__(self):
        super().__init__()
        self.recognizer = SignStreamRecognizer()

    def run(self):
        if cv2 is None or mp is None:
            print("Camera dependencies unavailable (cv2/mediapipe)")
            return
        from core.vision.pose_adapter import mediapipe_to_pose_dict

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        mp_holistic = mp.solutions.holistic.Holistic(
            model_complexity=0,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        thread = QThread.currentThread()

        while cap.isOpened() and not thread.isInterruptionRequested():
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_holistic.process(rgb)

            # Extract all landmark groups safely
            pose_landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
            left_hand_landmarks = result.left_hand_landmarks.landmark if result.left_hand_landmarks else None
            right_hand_landmarks = result.right_hand_landmarks.landmark if result.right_hand_landmarks else None

            if pose_landmarks:
                pose = mediapipe_to_pose_dict(
                    pose_landmarks,
                    left_hand_landmarks,
                    right_hand_landmarks
                )
                self.pose_ready.emit(pose)
                update = self.recognizer.process(pose)
                if update.detected_token is not None:
                    self.token_ready.emit(update.detected_token, update.confidence)
                if update.sentence_tokens is not None:
                    self.sequence_ready.emit(update.sentence_tokens)

        cap.release()
        print("Camera thread exiting cleanly")
