from typing import Dict, Tuple
from core.skeleton.canonical import CanonicalSkeleton

Vec2 = Tuple[float, float]

FINGER_CHAINS = {
    "thumb":  [1, 2, 3, 4],
    "index":  [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring":   [13, 14, 15, 16],
    "pinky":  [17, 18, 19, 20],
}


def _xy(v):
    # Flip X to match animation coordinate system
    return (-v[0], v[1])


def to_animation_joints(skel: CanonicalSkeleton) -> Dict[str, Vec2]:
    joints = {}

    def xy(v):
        return (-v[0], v[1])  # handedness already fixed

    # --- arms ---
    if skel.left_arm:
        joints["left_shoulder"] = xy(skel.left_arm.shoulder)
        joints["left_elbow"] = xy(skel.left_arm.elbow)
        joints["left_wrist"] = xy(skel.left_arm.wrist)

    if skel.right_arm:
        joints["right_shoulder"] = xy(skel.right_arm.shoulder)
        joints["right_elbow"] = xy(skel.right_arm.elbow)
        joints["right_wrist"] = xy(skel.right_arm.wrist)

    # --- left hand fingers ---
    if skel.left_hand:
        for finger, indices in FINGER_CHAINS.items():
            for i, idx in enumerate(indices):
                name = f"left_{finger}_{i}"
                joints[name] = xy(skel.left_hand.landmarks[idx])

    # --- right hand fingers ---
    if skel.right_hand:
        for finger, indices in FINGER_CHAINS.items():
            for i, idx in enumerate(indices):
                name = f"right_{finger}_{i}"
                joints[name] = xy(skel.right_hand.landmarks[idx])

    return joints

