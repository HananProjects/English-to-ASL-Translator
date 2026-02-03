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

    results = detector.process(frame)
    skel = normalize(from_mediapipe(results))
    joints = to_animation_joints(skel)

    for name, (x, y) in joints.items():
        print(name, (round(x, 2), round(y, 2)))

    cv2.imshow("Animation Mapping Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
