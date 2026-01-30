import mediapipe as mp

PL = mp.solutions.pose.PoseLandmark


def mediapipe_to_pose_dict(landmarks):
    def pt(i):
        return landmarks[i].x, landmarks[i].y

    # Anchor at torso (left hip)
    torso_x, torso_y = pt(PL.LEFT_HIP)

    def rel(i):
        x, y = pt(i)
        return (x - torso_x + 0.5, y - torso_y + 0.5)

    return {
        "head": rel(PL.NOSE),

        "shoulder_left": rel(PL.LEFT_SHOULDER),
        "elbow_left": rel(PL.LEFT_ELBOW),
        "hand_left": rel(PL.LEFT_WRIST),

        "shoulder_right": rel(PL.RIGHT_SHOULDER),
        "elbow_right": rel(PL.RIGHT_ELBOW),
        "hand_right": rel(PL.RIGHT_WRIST),

        "torso": (0.5, 0.5),
    }