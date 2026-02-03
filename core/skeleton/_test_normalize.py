import cv2
from core.vision.mediapipe_holistic import MediaPipeHolistic
from core.skeleton.canonical import from_mediapipe
from core.skeleton.normalize import normalize

detector = MediaPipeHolistic()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = detector.process(frame)
    skel = from_mediapipe(results)
    norm_skel = normalize(skel)

    if norm_skel.left_arm:
        print("Normalized left wrist:", norm_skel.left_arm.wrist)

    if norm_skel.left_hand:
        print("Normalized left hand joints:", len(norm_skel.left_hand.landmarks))

    cv2.imshow("Normalize Test (Visual Only)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
