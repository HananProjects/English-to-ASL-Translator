import mediapipe as mp
HL = mp.solutions.hands.HandLandmark

def mediapipe_hand_to_dict(landmarks, prefix):
    def pt(i):
        lm = landmarks.landmark[i]
        return lm.x, lm.y

    return {
        # Wrist
        f"{prefix}_wrist": pt(HL.WRIST),

        # Thumb
        f"{prefix}_thumb_mcp": pt(HL.THUMB_CMC),
        f"{prefix}_thumb_pip": pt(HL.THUMB_MCP),
        f"{prefix}_thumb_dip": pt(HL.THUMB_IP),
        f"{prefix}_thumb_tip": pt(HL.THUMB_TIP),

        # Index
        f"{prefix}_index_mcp": pt(HL.INDEX_FINGER_MCP),
        f"{prefix}_index_pip": pt(HL.INDEX_FINGER_PIP),
        f"{prefix}_index_dip": pt(HL.INDEX_FINGER_DIP),
        f"{prefix}_index_tip": pt(HL.INDEX_FINGER_TIP),

        # Middle
        f"{prefix}_middle_mcp": pt(HL.MIDDLE_FINGER_MCP),
        f"{prefix}_middle_pip": pt(HL.MIDDLE_FINGER_PIP),
        f"{prefix}_middle_dip": pt(HL.MIDDLE_FINGER_DIP),
        f"{prefix}_middle_tip": pt(HL.MIDDLE_FINGER_TIP),

        # Ring
        f"{prefix}_ring_mcp": pt(HL.RING_FINGER_MCP),
        f"{prefix}_ring_pip": pt(HL.RING_FINGER_PIP),
        f"{prefix}_ring_dip": pt(HL.RING_FINGER_DIP),
        f"{prefix}_ring_tip": pt(HL.RING_FINGER_TIP),

        # Pinky
        f"{prefix}_pinky_mcp": pt(HL.PINKY_MCP),
        f"{prefix}_pinky_pip": pt(HL.PINKY_PIP),
        f"{prefix}_pinky_dip": pt(HL.PINKY_DIP),
        f"{prefix}_pinky_tip": pt(HL.PINKY_TIP),
    }