from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

Vec3 = Tuple[float, float, float]


@dataclass
class Arm:
    shoulder: Vec3
    elbow: Vec3
    wrist: Vec3


@dataclass
class Hand:
    landmarks: List[Vec3]  # 21 points, MediaPipe order


@dataclass
class CanonicalSkeleton:
    left_arm: Optional[Arm]
    right_arm: Optional[Arm]
    left_hand: Optional[Hand]
    right_hand: Optional[Hand]


def _vec3(lm) -> Vec3:
    return (lm.x, lm.y, lm.z)


def from_mediapipe(results) -> CanonicalSkeleton:
    left_arm = right_arm = None
    left_hand = right_hand = None

    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark

        left_arm = Arm(
            shoulder=_vec3(pose[11]),  # LEFT_SHOULDER
            elbow=_vec3(pose[13]),     # LEFT_ELBOW
            wrist=_vec3(pose[15])      # LEFT_WRIST
        )

        right_arm = Arm(
            shoulder=_vec3(pose[12]),  # RIGHT_SHOULDER
            elbow=_vec3(pose[14]),     # RIGHT_ELBOW
            wrist=_vec3(pose[16])      # RIGHT_WRIST
        )

    if results.left_hand_landmarks:
        left_hand = Hand(
            landmarks=[_vec3(lm) for lm in results.left_hand_landmarks.landmark]
        )

    if results.right_hand_landmarks:
        right_hand = Hand(
            landmarks=[_vec3(lm) for lm in results.right_hand_landmarks.landmark]
        )

    return CanonicalSkeleton(
        left_arm=left_arm,
        right_arm=right_arm,
        left_hand=left_hand,
        right_hand=right_hand
    )
