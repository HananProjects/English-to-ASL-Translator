import cv2
from core.vision.mediapipe_holistic import MediaPipeHolistic
from core.skeleton.canonical import from_mediapipe
from core.skeleton.normalize import normalize
from core.skeleton.to_animation import to_animation_joints
from core.skeleton.smooth import EMASmoother

detector = MediaPipeHolistic()
cap = cv2.VideoCapture(0)
smoother = EMASmoother(alpha=0.4)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    skel = normalize(from_mediapipe(detector.process(frame)))
    joints = to_animation_joints(skel)
    smooth_joints = smoother.update(joints)

    # Compare one joint
    if "left_wrist" in joints:
        print(
            "raw:", tuple(round(v, 3) for v in joints["left_wrist"]),
            "smooth:", tuple(round(v, 3) for v in smooth_joints["left_wrist"])
        )

    cv2.imshow("Smoothing Test", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
