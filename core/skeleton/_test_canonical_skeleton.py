import cv2
from core.vision.camera import Camera
from core.vision.mediapipe_holistic import MediaPipeHolistic
from core.skeleton.canonical import from_mediapipe

# --- setup ---
cam = Camera()
detector = MediaPipeHolistic()

while True:
    frame = cam.read()
    if frame is None:
        break

    results = detector.process(frame)
    skeleton = from_mediapipe(results)

    # --- sanity prints ---
    if skeleton.left_arm:
        print("Left wrist:", skeleton.left_arm.wrist)

    if skeleton.left_hand:
        print("Left hand landmarks:", len(skeleton.left_hand.landmarks))

    cv2.imshow("Canonical Skeleton Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
