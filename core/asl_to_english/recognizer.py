import json
import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

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

HAND_KEYS = (
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

LEFT_HAND_KEYS = (
    "left_thumb_tip",
    "left_index_tip",
    "left_middle_tip",
    "left_ring_tip",
    "left_pinky_tip",
)

RIGHT_HAND_KEYS = (
    "right_thumb_tip",
    "right_index_tip",
    "right_middle_tip",
    "right_ring_tip",
    "right_pinky_tip",
)

LEFT_ARM_KEYS = (
    "shoulder_left",
    "elbow_left",
    "hand_left",
)

RIGHT_ARM_KEYS = (
    "shoulder_right",
    "elbow_right",
    "hand_right",
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
    raw_token: Optional[str] = None
    raw_confidence: float = 0.0
    candidate_streak: int = 0


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


def pose_to_feature_vector(pose: PoseDict) -> np.ndarray:
    normalized = normalize_pose(pose)
    feat: List[float] = []
    for key in JOINT_KEYS:
        point = normalized.get(key)
        if point is None:
            feat.extend([0.0, 0.0])
        else:
            feat.extend([float(point[0]), float(point[1])])
    vector = np.asarray(feat, dtype=np.float32)
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(vector, -20.0, 20.0)


def suppress_inactive_hand_noise(pose: PoseDict) -> PoseDict:
    """
    If only one hand has reliable fingertip landmarks in this frame,
    drop the opposite arm/hand keys to reduce one-hand sign confusion.
    """
    if not pose:
        return pose

    left_count = sum(1 for key in LEFT_HAND_KEYS if key in pose)
    right_count = sum(1 for key in RIGHT_HAND_KEYS if key in pose)
    min_hand_points = 2
    strong_hand_points = 4

    # Both hands present (or both absent): leave pose unchanged.
    if left_count >= min_hand_points and right_count >= min_hand_points:
        # If one side has much weaker fingertip evidence, suppress it.
        if left_count >= strong_hand_points and right_count <= 2:
            filtered = dict(pose)
            for key in RIGHT_HAND_KEYS + RIGHT_ARM_KEYS:
                filtered.pop(key, None)
            return filtered
        if right_count >= strong_hand_points and left_count <= 2:
            filtered = dict(pose)
            for key in LEFT_HAND_KEYS + LEFT_ARM_KEYS:
                filtered.pop(key, None)
            return filtered
        return pose

    if left_count < min_hand_points and right_count < min_hand_points:
        return pose

    filtered = dict(pose)
    if left_count >= min_hand_points and right_count < min_hand_points:
        for key in RIGHT_HAND_KEYS + RIGHT_ARM_KEYS:
            filtered.pop(key, None)
    elif right_count >= min_hand_points and left_count < min_hand_points:
        for key in LEFT_HAND_KEYS + LEFT_ARM_KEYS:
            filtered.pop(key, None)
    return filtered


def _hand_motion(pose_a: PoseDict, pose_b: PoseDict, side: str) -> float:
    if side == "left":
        keys = ("hand_left",) + LEFT_HAND_KEYS
    else:
        keys = ("hand_right",) + RIGHT_HAND_KEYS
    dists = []
    for key in keys:
        a = pose_a.get(key)
        b = pose_b.get(key)
        if a is None or b is None:
            continue
        dists.append(math.dist(a, b))
    if not dists:
        return 0.0
    return float(sum(dists) / len(dists))


def pose_distance(a: PoseDict, b: PoseDict, min_shared: int = 9) -> float:
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
        if len(pose_vector) < 8 or not self.templates:
            return None, 0.0

        # Reject low-information frames (hands mostly missing), which cause
        # unstable nearest-template jumps.
        hand_points = sum(1 for key in HAND_KEYS if key in pose_vector)
        if hand_points < 2:
            return None, 0.0

        best_token = None
        best_distance = float("inf")
        second_best_distance = float("inf")

        for template in self.templates:
            distance = pose_distance(pose_vector, template.vector)
            if distance < best_distance:
                second_best_distance = best_distance
                best_distance = distance
                best_token = template.token
            elif distance < second_best_distance:
                second_best_distance = distance

        if best_token is None or not math.isfinite(best_distance):
            return None, 0.0

        # Absolute and relative checks to suppress ambiguous matches.
        if best_distance > 0.90:
            return None, 0.0
        if math.isfinite(second_best_distance) and (second_best_distance - best_distance) < 0.005:
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


class ModelMatcher:
    def __init__(self, model_path: Path):
        model = np.load(model_path, allow_pickle=True)
        self.W = np.nan_to_num(model["W"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.b = np.nan_to_num(model["b"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.mean = np.nan_to_num(model["mean"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        self.std = np.nan_to_num(model["std"].astype(np.float32), nan=1.0, posinf=1.0, neginf=1.0)
        self.std[self.std < 1e-6] = 1.0
        self.W = np.clip(self.W, -10.0, 10.0)
        self.b = np.clip(self.b, -10.0, 10.0)
        self.mean = np.clip(self.mean, -20.0, 20.0)
        self.std = np.clip(self.std, 1e-3, 1000.0)
        labels_raw = model["labels"]
        self.labels = [str(v) for v in labels_raw.tolist()]
        self.seq_len = int(model["seq_len"])
        self.feature_dim = int(model["feature_dim"])
        self._buffer: List[np.ndarray] = []

        if self.W.ndim != 2 or self.b.ndim != 1:
            raise ValueError("Invalid model parameter shapes")
        if self.W.shape[1] != len(self.labels) or self.b.shape[0] != len(self.labels):
            raise ValueError("Model output dimension does not match labels")
        if self.mean.shape[1] != self.W.shape[0] or self.std.shape[1] != self.W.shape[0]:
            raise ValueError("Feature normalization shape mismatch")
        if self.seq_len * self.feature_dim != self.W.shape[0]:
            raise ValueError("Model feature size does not match seq_len/feature_dim")

    @staticmethod
    def default_model_path() -> Path:
        repo_root = Path(__file__).resolve().parents[2]
        v2 = repo_root / "models" / "asl_landmark_classifier_v2.npz"
        if v2.exists():
            return v2
        return repo_root / "models" / "asl_landmark_classifier_v1.npz"

    @classmethod
    def try_create(cls, model_path: Optional[Path] = None) -> Optional["ModelMatcher"]:
        path = model_path or cls.default_model_path()
        if not path.exists():
            return None
        try:
            return cls(path)
        except Exception:
            return None

    def match(self, pose: PoseDict) -> Tuple[Optional[str], float]:
        frame_feat = pose_to_feature_vector(pose)
        if frame_feat.shape[0] != self.feature_dim:
            return None, 0.0

        self._buffer.append(frame_feat)
        if len(self._buffer) > self.seq_len:
            self._buffer = self._buffer[-self.seq_len:]
        if len(self._buffer) < self.seq_len:
            return None, 0.0

        seq = np.stack(self._buffer, axis=0).reshape(1, -1)
        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
        seq = np.clip(seq, -8.0, 8.0)
        X = (seq - self.mean) / self.std
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        X = np.clip(X, -8.0, 8.0)

        # Use float64 for the matmul to avoid sporadic float32 backend warnings
        # on some platforms, then clamp back to a stable range.
        logits = X.astype(np.float64) @ self.W.astype(np.float64)
        logits = logits + self.b.astype(np.float64)
        logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        logits = np.clip(logits, -60.0, 60.0)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-9)

        best_idx = int(np.argmax(probs[0]))
        confidence = float(probs[0, best_idx])
        if best_idx < 0 or best_idx >= len(self.labels):
            return None, 0.0

        token = self.labels[best_idx].upper()
        return token, confidence

    def reset(self):
        self._buffer.clear()


class HybridMatcher:
    """
    Combines model and template matchers to reduce obvious mislabels.
    """

    def __init__(
        self,
        model_matcher: Optional[ModelMatcher],
        template_matcher: Optional[ClipTemplateMatcher] = None,
        strong_model_conf: float = 0.92,
        strong_template_conf: float = 0.82,
    ):
        self.model_matcher = model_matcher
        self.template_matcher = template_matcher or ClipTemplateMatcher()
        self.strong_model_conf = strong_model_conf
        self.strong_template_conf = strong_template_conf
        self.disagreement_margin = 0.12
        self.min_disagreement_conf = 0.62

    def match(self, pose: PoseDict) -> Tuple[Optional[str], float]:
        template_token, template_conf = self.template_matcher.match(pose)
        if self.model_matcher is None:
            return template_token, template_conf

        model_token, model_conf = self.model_matcher.match(pose)

        if model_token is None:
            return template_token, template_conf
        if template_token is None:
            return model_token, model_conf

        # When both agree, keep the label and trust the stronger confidence.
        if model_token == template_token:
            return model_token, max(model_conf, template_conf)

        # If one matcher is very confident while the other is not, trust it.
        if model_conf >= self.strong_model_conf and template_conf < 0.60:
            return model_token, model_conf
        if template_conf >= self.strong_template_conf and model_conf < 0.75:
            return template_token, template_conf

        # If one matcher is clearly stronger, prefer it instead of dropping frames.
        if (
            model_conf >= self.min_disagreement_conf
            and (model_conf - template_conf) >= self.disagreement_margin
        ):
            return model_token, model_conf
        if (
            template_conf >= self.min_disagreement_conf
            and (template_conf - model_conf) >= self.disagreement_margin
        ):
            return template_token, template_conf

        # Ambiguous disagreement: reject this frame to avoid wrong token commits.
        return None, 0.0

    def reset(self):
        if self.model_matcher is not None:
            self.model_matcher.reset()


def build_default_matcher(
    prefer_model: bool = True,
    model_path: Optional[Path] = None,
):
    if prefer_model:
        model_matcher = ModelMatcher.try_create(model_path=model_path)
        if model_matcher is not None:
            return HybridMatcher(model_matcher=model_matcher)
    return ClipTemplateMatcher()


class SignStreamRecognizer:
    def __init__(
        self,
        matcher=None,
        prefer_model: bool = True,
        model_path: Optional[Path] = None,
        stable_frames: int = 8,
        min_confidence: float = 0.70,
        emit_cooldown_frames: int = 12,
        pause_frames: int = 20,
    ):
        self.matcher = matcher or build_default_matcher(
            prefer_model=prefer_model,
            model_path=model_path,
        )
        self.stable_frames = max(1, stable_frames)
        self.min_confidence = min_confidence
        self.emit_cooldown_frames = max(0, emit_cooldown_frames)
        self.pause_frames = max(1, pause_frames)
        # Hard safety gate for token commits during live recognition.
        self.commit_min_confidence = 0.75

        self.buffered_tokens: List[str] = []
        self._candidate_token: Optional[str] = None
        self._candidate_streak = 0
        self._last_emitted_token: Optional[str] = None
        self._cooldown_left = 0
        self._low_conf_streak = 0
        self._right_hand_history = deque(maxlen=12)
        self._hello_motion_cooldown = 0
        self._prev_pose: Optional[PoseDict] = None
        self._left_motion_history = deque(maxlen=8)
        self._right_motion_history = deque(maxlen=8)
        self._suppress_idle_hand = os.getenv("ASL_SUPPRESS_IDLE_HAND", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def process(self, pose: PoseDict) -> RecognitionUpdate:
        if self._prev_pose is not None:
            self._left_motion_history.append(_hand_motion(pose, self._prev_pose, "left"))
            self._right_motion_history.append(_hand_motion(pose, self._prev_pose, "right"))
        self._prev_pose = pose

        filtered_pose = suppress_inactive_hand_noise(pose)
        # Keep both hands by default; enabling suppression can hurt two-hand signs.
        if self._suppress_idle_hand:
            filtered_pose = self._suppress_idle_opposite_hand(filtered_pose)
        token, confidence = self.matcher.match(filtered_pose)
        motion_token, motion_conf = self._match_hello_motion(filtered_pose)
        if motion_token is not None:
            token, confidence = motion_token, motion_conf
        detected_token = None
        sentence_tokens = None

        if self._cooldown_left > 0:
            self._cooldown_left -= 1
        if self._hello_motion_cooldown > 0:
            self._hello_motion_cooldown -= 1

        effective_min_conf = max(self.min_confidence, self.commit_min_confidence)
        if token is None or confidence < effective_min_conf:
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
                raw_token=token,
                raw_confidence=confidence,
                candidate_streak=self._candidate_streak,
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
            raw_token=token,
            raw_confidence=confidence,
            candidate_streak=self._candidate_streak,
        )

    def reset(self):
        self.buffered_tokens.clear()
        self._candidate_token = None
        self._candidate_streak = 0
        self._last_emitted_token = None
        self._cooldown_left = 0
        self._low_conf_streak = 0
        self._right_hand_history.clear()
        self._hello_motion_cooldown = 0
        self._prev_pose = None
        self._left_motion_history.clear()
        self._right_motion_history.clear()
        reset_fn = getattr(self.matcher, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def _suppress_idle_opposite_hand(self, pose: PoseDict) -> PoseDict:
        left_points = sum(1 for key in LEFT_HAND_KEYS if key in pose)
        right_points = sum(1 for key in RIGHT_HAND_KEYS if key in pose)
        if left_points < 2 or right_points < 2:
            return pose
        if not self._left_motion_history or not self._right_motion_history:
            return pose

        left_motion = sum(self._left_motion_history) / len(self._left_motion_history)
        right_motion = sum(self._right_motion_history) / len(self._right_motion_history)

        active_thresh = 0.010
        idle_thresh = 0.0035
        filtered = dict(pose)
        if left_motion > active_thresh and right_motion < idle_thresh:
            for key in RIGHT_HAND_KEYS + RIGHT_ARM_KEYS:
                filtered.pop(key, None)
            return filtered
        if right_motion > active_thresh and left_motion < idle_thresh:
            for key in LEFT_HAND_KEYS + LEFT_ARM_KEYS:
                filtered.pop(key, None)
            return filtered
        return pose

    def _match_hello_motion(self, pose: PoseDict) -> Tuple[Optional[str], float]:
        """
        HELLO is a movement sign; single-frame matching often confuses it with KNOW.
        Detect a short right-hand trajectory from forehead area outward.
        """
        if self._hello_motion_cooldown > 0:
            return None, 0.0

        head = pose.get("head")
        hand = pose.get("hand_right")
        if head is None or hand is None:
            return None, 0.0

        self._right_hand_history.append((hand[0], hand[1], head[0], head[1]))
        if len(self._right_hand_history) < 8:
            return None, 0.0

        start = self._right_hand_history[0]
        end = self._right_hand_history[-1]
        sx, sy, shx, shy = start
        ex, ey, ehx, ehy = end

        # Start near forehead.
        start_dist = math.dist((sx, sy), (shx, shy))
        near_forehead = start_dist < 0.16 and abs(sy - shy) < 0.12

        # Move outward mainly in x (camera mirror can flip sign of delta).
        dx = ex - sx
        dy = ey - sy
        moved_outward = abs(dx) > 0.10 and abs(dy) < 0.12

        if near_forehead and moved_outward:
            self._hello_motion_cooldown = 20
            self._right_hand_history.clear()
            return "HELLO", 0.92

        return None, 0.0
