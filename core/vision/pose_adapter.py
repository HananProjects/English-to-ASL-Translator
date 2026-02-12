import mediapipe as mp

PL = mp.solutions.pose.PoseLandmark


def mediapipe_to_pose_dict(
    pose_landmarks,
    left_hand_landmarks=None,
    right_hand_landmarks=None
):
    def pt_pose(i):
        return pose_landmarks[i].x, pose_landmarks[i].y

    def norm(x, y):
        # MediaPipe already provides normalized image coordinates.
        x_n = max(0.0, min(1.0, x))
        y_n = max(0.0, min(1.0, y))
        return (x_n, y_n)

    pose = {
        "head": norm(*pt_pose(PL.NOSE)),

        "shoulder_left": norm(*pt_pose(PL.LEFT_SHOULDER)),
        "elbow_left": norm(*pt_pose(PL.LEFT_ELBOW)),
        "hand_left": norm(*pt_pose(PL.LEFT_WRIST)),

        "shoulder_right": norm(*pt_pose(PL.RIGHT_SHOULDER)),
        "elbow_right": norm(*pt_pose(PL.RIGHT_ELBOW)),
        "hand_right": norm(*pt_pose(PL.RIGHT_WRIST)),

        "torso": norm(*pt_pose(PL.LEFT_HIP)),
    }

    # --- LEFT HAND ---
    if left_hand_landmarks:
        for idx, lm in enumerate(left_hand_landmarks):
            pose[f"left_hand_{idx}"] = norm(lm.x, lm.y)

        pose["left_thumb_tip"] = norm(
            left_hand_landmarks[4].x,
            left_hand_landmarks[4].y
        )
        pose["left_index_tip"] = norm(
            left_hand_landmarks[8].x,
            left_hand_landmarks[8].y
        )
        pose["left_middle_tip"] = norm(
            left_hand_landmarks[12].x,
            left_hand_landmarks[12].y
        )
        pose["left_ring_tip"] = norm(
            left_hand_landmarks[16].x,
            left_hand_landmarks[16].y
        )
        pose["left_pinky_tip"] = norm(
            left_hand_landmarks[20].x,
            left_hand_landmarks[20].y
        )

    # --- RIGHT HAND ---
    if right_hand_landmarks:
        for idx, lm in enumerate(right_hand_landmarks):
            pose[f"right_hand_{idx}"] = norm(lm.x, lm.y)

        pose["right_thumb_tip"] = norm(
            right_hand_landmarks[4].x,
            right_hand_landmarks[4].y
        )
        pose["right_index_tip"] = norm(
            right_hand_landmarks[8].x,
            right_hand_landmarks[8].y
        )
        pose["right_middle_tip"] = norm(
            right_hand_landmarks[12].x,
            right_hand_landmarks[12].y
        )
        pose["right_ring_tip"] = norm(
            right_hand_landmarks[16].x,
            right_hand_landmarks[16].y
        )
        pose["right_pinky_tip"] = norm(
            right_hand_landmarks[20].x,
            right_hand_landmarks[20].y
        )

    return pose
