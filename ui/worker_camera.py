import cv2
import mediapipe as mp
from PySide6.QtCore import QObject, Signal, QThread
from core.vision.pose_adapter import mediapipe_to_pose_dict

class CameraWorker(QObject):
    pose_ready = Signal(dict)

    def run(self):
        cap = cv2.VideoCapture(0)
        mp_pose = mp.solutions.pose.Pose()

        thread = QThread.currentThread()

        while cap.isOpened() and not thread.isInterruptionRequested():
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = mp_pose.process(rgb)

            if result.pose_landmarks:
                print("POSE DETECTED")
                pose = mediapipe_to_pose_dict(
                    result.pose_landmarks.landmark
                )
                self.pose_ready.emit(pose)

        cap.release()
        print("Camera thread exiting cleanly")