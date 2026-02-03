from typing import Tuple
from core.skeleton.canonical import CanonicalSkeleton, Arm, Hand

Vec3 = Tuple[float, float, float]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _div(v: Vec3, s: float) -> Vec3:
    return (v[0] / s, v[1] / s, v[2] / s)


def _dist(a: Vec3, b: Vec3) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def normalize(skel: CanonicalSkeleton) -> CanonicalSkeleton:
    # --- must have both shoulders ---
    if not skel.left_arm or not skel.right_arm:
        return skel

    left_sh = skel.left_arm.shoulder
    right_sh = skel.right_arm.shoulder

    # --- anchor: shoulder midpoint ---
    anchor = (
        (left_sh[0] + right_sh[0]) / 2,
        (left_sh[1] + right_sh[1]) / 2,
        (left_sh[2] + right_sh[2]) / 2,
    )

    # --- scale: shoulder width ---
    scale = _dist(left_sh, right_sh)
    if scale < 1e-6:
        return skel  # avoid divide-by-zero

    def norm(v: Vec3) -> Vec3:
        return _div(_sub(v, anchor), scale)

    # --- normalize arms ---
    left_arm = Arm(
        shoulder=norm(skel.left_arm.shoulder),
        elbow=norm(skel.left_arm.elbow),
        wrist=norm(skel.left_arm.wrist),
    ) if skel.left_arm else None

    right_arm = Arm(
        shoulder=norm(skel.right_arm.shoulder),
        elbow=norm(skel.right_arm.elbow),
        wrist=norm(skel.right_arm.wrist),
    ) if skel.right_arm else None

    # --- normalize hands ---
    left_hand = (
        Hand([norm(v) for v in skel.left_hand.landmarks])
        if skel.left_hand else None
    )

    right_hand = (
        Hand([norm(v) for v in skel.right_hand.landmarks])
        if skel.right_hand else None
    )

    return CanonicalSkeleton(
        left_arm=left_arm,
        right_arm=right_arm,
        left_hand=left_hand,
        right_hand=right_hand
    )
