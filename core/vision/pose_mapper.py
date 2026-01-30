def mediapipe_to_skeleton(landmarks):
    """
    Convert MediaPipe landmarks to normalized skeleton dict
    Coordinates are normalized [0,1]
    """

    def lm(i):
        return landmarks[i].x, landmarks[i].y

    return {
        "head": lm(0),
        "shoulder": lm(12),
        "elbow_right": lm(14),
        "hand_right": lm(16),
        "elbow_left": lm(13),
        "hand_left": lm(15),
        "torso": lm(24),
    }