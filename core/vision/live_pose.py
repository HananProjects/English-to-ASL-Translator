from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

PoseDict = Dict[str, Tuple[float, float]]

HAND_RELATED_KEYS = {
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
}

BODY_RELATED_KEYS = {
    "head",
    "shoulder_left",
    "elbow_left",
    "shoulder_right",
    "elbow_right",
    "torso",
}


@dataclass
class LivePoseFilter:
    alpha: float = 0.55
    hand_hold_frames: int = 3
    body_hold_frames: int = 1
    _state: PoseDict = field(default_factory=dict)
    _missing_counts: Dict[str, int] = field(default_factory=dict)

    def apply(self, pose: PoseDict) -> PoseDict:
        if not pose:
            self._advance_missing(self._state.keys())
            return {}

        output: PoseDict = {}
        seen = set()

        for key, point in pose.items():
            prev = self._state.get(key)
            if prev is None:
                smoothed = (float(point[0]), float(point[1]))
            else:
                smoothed = (
                    (prev[0] * (1.0 - self.alpha)) + (float(point[0]) * self.alpha),
                    (prev[1] * (1.0 - self.alpha)) + (float(point[1]) * self.alpha),
                )
            output[key] = smoothed
            self._state[key] = smoothed
            self._missing_counts[key] = 0
            seen.add(key)

        for key in list(self._state.keys()):
            if key in seen:
                continue
            hold_limit = self._hold_limit(key)
            if hold_limit <= 0:
                continue
            missed = self._missing_counts.get(key, 0) + 1
            self._missing_counts[key] = missed
            if missed <= hold_limit:
                output[key] = self._state[key]

        self._prune_expired()
        return output

    def reset(self) -> None:
        self._state.clear()
        self._missing_counts.clear()

    def _advance_missing(self, keys: Iterable[str]) -> None:
        for key in keys:
            self._missing_counts[key] = self._missing_counts.get(key, 0) + 1
        self._prune_expired()

    def _prune_expired(self) -> None:
        for key in list(self._state.keys()):
            hold_limit = self._hold_limit(key)
            if hold_limit <= 0:
                self._state.pop(key, None)
                self._missing_counts.pop(key, None)
                continue
            if self._missing_counts.get(key, 0) > hold_limit:
                self._state.pop(key, None)
                self._missing_counts.pop(key, None)

    def _hold_limit(self, key: str) -> int:
        if key.startswith("left_hand_") or key.startswith("right_hand_"):
            return self.hand_hold_frames
        if key in HAND_RELATED_KEYS:
            return self.hand_hold_frames
        if key in BODY_RELATED_KEYS:
            return self.body_hold_frames
        return 0
