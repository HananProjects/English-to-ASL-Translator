import mediapipe as mp

HL = mp.solutions.hands.HandLandmark

def mediapipe_hand_to_dict(landmarks, prefix):
    def pt(i):
        lm = landmarks.landmark[i]
        return lm.x, lm.y

    return {
        f"{prefix}_wrist": pt(HL.WRIST),

        f"{prefix}_thumb_tip": pt(HL.THUMB_TIP),
        f"{prefix}_index_tip": pt(HL.INDEX_FINGER_TIP),
        f"{prefix}_middle_tip": pt(HL.MIDDLE_FINGER_TIP),
        f"{prefix}_ring_tip": pt(HL.RING_FINGER_TIP),
        f"{prefix}_pinky_tip": pt(HL.PINKY_TIP),
    }