import cv2
from core.vision.mediapipe_holistic import MediaPipeHolistic
from core.skeleton.canonical import from_mediapipe
from core.skeleton.normalize import normalize
from core.skeleton.to_animation import to_animation_joints

detector = MediaPipeHolistic()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    skel = normalize(from_mediapipe(detector.process(frame)))
    joints = to_animation_joints(skel)

    # sanity checks
    print("Total joints:", len(joints))
    print("Example:", list(joints.items())[:5])

    cv2.imshow("Hand Animation Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
