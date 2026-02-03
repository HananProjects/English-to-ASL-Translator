from typing import Dict, List, Tuple
import numpy as np

Vec2 = Tuple[float, float]

JOINT_ORDER: List[str] = [
    # arms
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder", "right_elbow", "right_wrist",

    # left hand
    "left_thumb_0", "left_thumb_1", "left_thumb_2", "left_thumb_3",
    "left_index_0", "left_index_1", "left_index_2", "left_index_3",
    "left_middle_0", "left_middle_1", "left_middle_2", "left_middle_3",
    "left_ring_0", "left_ring_1", "left_ring_2", "left_ring_3",
    "left_pinky_0", "left_pinky_1", "left_pinky_2", "left_pinky_3",

    # right hand
    "right_thumb_0", "right_thumb_1", "right_thumb_2", "right_thumb_3",
    "right_index_0", "right_index_1", "right_index_2", "right_index_3",
    "right_middle_0", "right_middle_1", "right_middle_2", "right_middle_3",
    "right_ring_0", "right_ring_1", "right_ring_2", "right_ring_3",
    "right_pinky_0", "right_pinky_1", "right_pinky_2", "right_pinky_3",
]


def joints_to_feature_vector(
    joints: Dict[str, Vec2],
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Convert smoothed animation joints into a flat ML feature vector.
    Output shape: (len(JOINT_ORDER) * 2,)
    """

    features: List[float] = []

    for name in JOINT_ORDER:
        if name in joints:
            x, y = joints[name]
            features.extend([x, y])
        else:
            # joint missing (hand out of frame, etc.)
            features.extend([fill_value, fill_value])

    return np.array(features, dtype=np.float32)
