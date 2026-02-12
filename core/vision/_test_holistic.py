import cv2
from core.vision.mediapipe_holistic import MediaPipeHolistic
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
detector = MediaPipeHolistic()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.process(frame)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp.solutions.holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp.solutions.holistic.HAND_CONNECTIONS
        )

    cv2.imshow("Holistic Sanity Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
