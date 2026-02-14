import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from core.english_to_asl.dictionary.asl_signs import ASL_SIGNS

PoseDict = Dict[str, Tuple[float, float]]

JOINT_KEYS = (
    "head",
    "shoulder_left",
    "elbow_left",
    "hand_left",
    "shoulder_right",
    "elbow_right",
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


@dataclass
class PoseTemplate:
    token: str
    vector: PoseDict


@dataclass
class RecognitionUpdate:
    detected_token: Optional[str]
    confidence: float
    sentence_tokens: Optional[List[str]]
    buffered_tokens: List[str]


def normalize_pose(pose: PoseDict) -> PoseDict:
    if not pose:
        return {}

    shoulder_left = pose.get("shoulder_left")
    shoulder_right = pose.get("shoulder_right")
    torso = pose.get("torso")

    if torso is None:
        if shoulder_left and shoulder_right:
            torso = (
                (shoulder_left[0] + shoulder_right[0]) / 2.0,
                (shoulder_left[1] + shoulder_right[1]) / 2.0,
            )
        elif shoulder_left is not None:
            torso = shoulder_left
        elif shoulder_right is not None:
            torso = shoulder_right
        else:
            torso = (0.5, 0.5)

    scale = 0.25
    if shoulder_left and shoulder_right:
        scale = math.dist(shoulder_left, shoulder_right)
        if scale < 1e-4:
            scale = 0.25

    normalized: PoseDict = {}
    for key in JOINT_KEYS:
        point = pose.get(key)
        if point is None:
            continue
        normalized[key] = (
            (point[0] - torso[0]) / scale,
            (point[1] - torso[1]) / scale,
        )
    return normalized


def pose_distance(a: PoseDict, b: PoseDict, min_shared: int = 6) -> float:
    shared = [key for key in a if key in b]
    if len(shared) < min_shared:
        return float("inf")
    total = 0.0
    for key in shared:
        total += math.dist(a[key], b[key])
    return total / len(shared)


class ClipTemplateMatcher:
    def __init__(
        self,
        clip_dir: Optional[Path] = None,
        sample_frames_per_clip: int = 5,
    ):
        if clip_dir is None:
            repo_root = Path(__file__).resolve().parents[2]
            clip_dir = repo_root / "ui" / "animation" / "clips"
        self.clip_dir = clip_dir
        self.sample_frames_per_clip = max(1, sample_frames_per_clip)
        self.templates: List[PoseTemplate] = self._build_templates()

    def match(self, pose: PoseDict) -> Tuple[Optional[str], float]:
        pose_vector = normalize_pose(pose)
        if len(pose_vector) < 6 or not self.templates:
            return None, 0.0

        best_token = None
        best_distance = float("inf")

        for template in self.templates:
            distance = pose_distance(pose_vector, template.vector)
            if distance < best_distance:
                best_distance = distance
                best_token = template.token

        if best_token is None or not math.isfinite(best_distance):
            return None, 0.0

        # Distance in normalized pose space is typically ~0..2 for useful matches.
        confidence = max(0.0, min(1.0, 1.0 - (best_distance / 2.0)))
        return best_token, confidence

    def _build_templates(self) -> List[PoseTemplate]:
        templates: List[PoseTemplate] = []
        for token, sign_def in ASL_SIGNS.items():
            clip_names: List[str] = []
            if isinstance(sign_def.get("clips"), list):
                clip_names.extend(sign_def["clips"])
            elif sign_def.get("clip"):
                clip_names.append(sign_def["clip"])

            for clip_name in clip_names:
                clip_path = self.clip_dir / (clip_name + ".json")
                if not clip_path.exists():
                    continue
                frames = self._load_clip_frames(clip_path)
                if not frames:
                    continue
                for idx in self._sample_indices(len(frames), self.sample_frames_per_clip):
                    vector = normalize_pose(frames[idx])
                    if len(vector) >= 6:
                        templates.append(PoseTemplate(token=token, vector=vector))
        return templates

    @staticmethod
    def _load_clip_frames(path: Path) -> List[PoseDict]:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                clip = json.load(handle)
            frames = clip.get("frames", [])
            if isinstance(frames, list):
                return frames
        except Exception:
            return []
        return []

    @staticmethod
    def _sample_indices(frame_count: int, sample_count: int) -> List[int]:
        if frame_count <= 0:
            return []
        if frame_count <= sample_count:
            return list(range(frame_count))
        if sample_count == 1:
            return [frame_count // 2]
        indices: List[int] = []
        step = (frame_count - 1) / float(sample_count - 1)
        for i in range(sample_count):
            indices.append(int(round(i * step)))
        return sorted(set(indices))


class SignStreamRecognizer:
    def __init__(
        self,
        matcher: Optional[ClipTemplateMatcher] = None,
        stable_frames: int = 4,
        min_confidence: float = 0.55,
        emit_cooldown_frames: int = 6,
        pause_frames: int = 12,
    ):
        self.matcher = matcher or ClipTemplateMatcher()
        self.stable_frames = max(1, stable_frames)
        self.min_confidence = min_confidence
        self.emit_cooldown_frames = max(0, emit_cooldown_frames)
        self.pause_frames = max(1, pause_frames)

        self.buffered_tokens: List[str] = []
        self._candidate_token: Optional[str] = None
        self._candidate_streak = 0
        self._last_emitted_token: Optional[str] = None
        self._cooldown_left = 0
        self._low_conf_streak = 0

    def process(self, pose: PoseDict) -> RecognitionUpdate:
        token, confidence = self.matcher.match(pose)
        detected_token = None
        sentence_tokens = None

        if self._cooldown_left > 0:
            self._cooldown_left -= 1

        if token is None or confidence < self.min_confidence:
            self._candidate_token = None
            self._candidate_streak = 0
            self._low_conf_streak += 1

            if self._low_conf_streak >= self.pause_frames and self.buffered_tokens:
                sentence_tokens = list(self.buffered_tokens)
                self.buffered_tokens.clear()
                self._last_emitted_token = None
                self._cooldown_left = 0

            return RecognitionUpdate(
                detected_token=None,
                confidence=confidence,
                sentence_tokens=sentence_tokens,
                buffered_tokens=list(self.buffered_tokens),
            )

        self._low_conf_streak = 0
        if token == self._candidate_token:
            self._candidate_streak += 1
        else:
            self._candidate_token = token
            self._candidate_streak = 1

        if (
            self._candidate_streak >= self.stable_frames
            and self._cooldown_left == 0
            and token != self._last_emitted_token
        ):
            self.buffered_tokens.append(token)
            detected_token = token
            self._last_emitted_token = token
            self._cooldown_left = self.emit_cooldown_frames

        return RecognitionUpdate(
            detected_token=detected_token,
            confidence=confidence,
            sentence_tokens=None,
            buffered_tokens=list(self.buffered_tokens),
        )
