from typing import Dict, Tuple
from core.skeleton.canonical import CanonicalSkeleton

Vec2 = Tuple[float, float]


def _xy(v):
    # Flip X to match animation coordinate system
    return (-v[0], v[1])


def to_animation_joints(skel: CanonicalSkeleton) -> Dict[str, Vec2]:
    """
    Convert a normalized skeleton into animation joint positions.
    Output is animation-space (2D), body-relative, unit scale.
    """
    joints = {}

    if skel.left_arm:
        joints["left_shoulder"] = _xy(skel.left_arm.shoulder)
        joints["left_elbow"] = _xy(skel.left_arm.elbow)
        joints["left_wrist"] = _xy(skel.left_arm.wrist)

    if skel.right_arm:
        joints["right_shoulder"] = _xy(skel.right_arm.shoulder)
        joints["right_elbow"] = _xy(skel.right_arm.elbow)
        joints["right_wrist"] = _xy(skel.right_arm.wrist)

    return joints
