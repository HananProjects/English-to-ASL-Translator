from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

PoseDict = Dict[str, Tuple[float, float]]

BODY_KEYS: Tuple[str, ...] = (
    "head",
    "shoulder_left",
    "elbow_left",
    "hand_left",
    "shoulder_right",
    "elbow_right",
    "hand_right",
)

LEFT_HAND_KEYS: Tuple[str, ...] = tuple(f"left_hand_{idx}" for idx in range(21))
RIGHT_HAND_KEYS: Tuple[str, ...] = tuple(f"right_hand_{idx}" for idx in range(21))
HAND_TIP_KEYS: Tuple[str, ...] = (
    "left_thumb_tip",
    "left_index_tip",
    "left_middle_tip",
    "left_ring_tip",
    "left_pinky_tip",
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_pinky_tip",
)

STATIC_POINT_KEYS: Tuple[str, ...] = BODY_KEYS + LEFT_HAND_KEYS + RIGHT_HAND_KEYS
MOTION_KEYS: Tuple[str, ...] = (
    "hand_left",
    "hand_right",
    "left_thumb_tip",
    "left_index_tip",
    "left_middle_tip",
    "left_ring_tip",
    "left_pinky_tip",
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_pinky_tip",
)


def _torso_and_scale(pose: PoseDict) -> Tuple[Tuple[float, float], float]:
    shoulder_left = pose.get("shoulder_left")
    shoulder_right = pose.get("shoulder_right")
    torso = pose.get("torso")
    if torso is None:
        if shoulder_left and shoulder_right:
            torso = (
                (shoulder_left[0] + shoulder_right[0]) / 2.0,
                (shoulder_left[1] + shoulder_right[1]) / 2.0,
            )
        elif shoulder_left:
            torso = shoulder_left
        elif shoulder_right:
            torso = shoulder_right
        else:
            torso = (0.5, 0.5)

    scale = 0.25
    if shoulder_left and shoulder_right:
        scale = math.dist(shoulder_left, shoulder_right)
        if scale < 1e-4:
            scale = 0.25
    return torso, scale


def normalize_pose_points(
    pose: PoseDict,
    keys: Tuple[str, ...] = STATIC_POINT_KEYS + HAND_TIP_KEYS,
) -> PoseDict:
    if not pose:
        return {}
    torso, scale = _torso_and_scale(pose)
    out: PoseDict = {}
    for key in keys:
        pt = pose.get(key)
        if pt is None:
            continue
        out[key] = ((pt[0] - torso[0]) / scale, (pt[1] - torso[1]) / scale)
    return out


def _safe_dist(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> float:
    if a is None or b is None:
        return 0.0
    return float(math.dist(a, b))


def _append_point_features(feat: List[float], pose: PoseDict, keys: Tuple[str, ...]) -> None:
    for key in keys:
        pt = pose.get(key)
        if pt is None:
            feat.extend([0.0, 0.0])
        else:
            feat.extend([float(pt[0]), float(pt[1])])


def _append_motion_features(
    feat: List[float],
    pose: PoseDict,
    prev_pose: Optional[PoseDict],
    keys: Tuple[str, ...],
) -> None:
    for key in keys:
        curr = pose.get(key)
        prev = prev_pose.get(key) if prev_pose else None
        if curr is None or prev is None:
            feat.extend([0.0, 0.0])
        else:
            feat.extend([float(curr[0] - prev[0]), float(curr[1] - prev[1])])


def _append_geometry_features(feat: List[float], pose: PoseDict) -> None:
    head = pose.get("head")
    left_hand = pose.get("hand_left")
    right_hand = pose.get("hand_right")
    shoulder_left = pose.get("shoulder_left")
    shoulder_right = pose.get("shoulder_right")
    torso = pose.get("torso")

    feat.extend(
        [
            _safe_dist(left_hand, head),
            _safe_dist(right_hand, head),
            _safe_dist(left_hand, torso),
            _safe_dist(right_hand, torso),
            _safe_dist(left_hand, right_hand),
            _safe_dist(shoulder_left, shoulder_right),
            _safe_dist(pose.get("left_thumb_tip"), pose.get("left_index_tip")),
            _safe_dist(pose.get("left_index_tip"), pose.get("left_pinky_tip")),
            _safe_dist(pose.get("right_thumb_tip"), pose.get("right_index_tip")),
            _safe_dist(pose.get("right_index_tip"), pose.get("right_pinky_tip")),
        ]
    )


def pose_to_feature_vector(pose: PoseDict, prev_pose: Optional[PoseDict] = None) -> np.ndarray:
    normalized = normalize_pose_points(pose)
    prev_normalized = normalize_pose_points(prev_pose) if prev_pose else None

    feat: List[float] = []
    _append_point_features(feat, normalized, STATIC_POINT_KEYS)
    _append_point_features(feat, normalized, HAND_TIP_KEYS)
    _append_motion_features(feat, normalized, prev_normalized, MOTION_KEYS)
    _append_geometry_features(feat, normalized)

    vector = np.asarray(feat, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(vector, -20.0, 20.0)
