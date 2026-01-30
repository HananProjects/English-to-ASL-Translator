import cv2
import mediapipe as mp
from PySide6.QtCore import QObject, Signal, QThread
from core.vision.pose_adapter import mediapipe_to_pose_dict
from core.vision.mediapipe_hands import MediaPipeHands
from core.vision.mediapipe_pose import MediaPipePose
from core.vision.hand_adapter import mediapipe_hand_to_dict

class CameraWorker(QObject):
    pose_ready = Signal(dict)

    def run(self):
        cap = cv2.VideoCapture(0)

        mp_pose = MediaPipePose()
        mp_hands = MediaPipeHands()

        thread = QThread.currentThread()

        while cap.isOpened() and not thread.isInterruptionRequested():
            ret, frame = cap.read()
            if not ret:
                continue

            # --- Pose ---
            pose_landmarks = mp_pose.process_frame(frame)
            if not pose_landmarks:
                continue

            print("POSE DETECTED")

            pose_dict = mediapipe_to_pose_dict(pose_landmarks)

            # --- Hands ---
            hand_landmarks_list = mp_hands.process_frame(frame)
            if hand_landmarks_list:
                for i, hand_landmarks in enumerate(hand_landmarks_list):
                    prefix = "left_hand" if i == 0 else "right_hand"
                    hand_dict = mediapipe_hand_to_dict(hand_landmarks, prefix)
                    pose_dict.update(hand_dict)

            # --- Emit once per frame ---
            self.pose_ready.emit(pose_dict)

        cap.release()
        print("Camera thread exiting cleanly")