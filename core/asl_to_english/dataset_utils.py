from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

PoseDict = Dict[str, Tuple[float, float]]

JOINT_KEYS: Tuple[str, ...] = (
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

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


@dataclass
class ClipSample:
    path: Path
    clip_name: str
    label: str
    signer_id: str
    session_id: str
    source: str
    split: Optional[str] = None
    num_frames: int = 0

    @property
    def group_key(self) -> str:
        if self.signer_id and self.signer_id != "unknown":
            return self.signer_id
        return self.clip_name


def normalize_id(value: str) -> str:
    value = _NON_ALNUM_RE.sub("_", value.strip().lower()).strip("_")
    return value or "unknown"


def infer_label_from_clip_name(stem: str) -> str:
    parts = stem.lower().split("_")
    out = []
    for part in parts:
        if part.isdigit():
            break
        out.append(part)
    if not out:
        out = [parts[0]]
    return "_".join(out).upper()


def infer_signer_from_clip_name(stem: str, label: str) -> str:
    parts = stem.split("_")
    label_parts = label.lower().split("_")
    if len(parts) >= len(label_parts) + 2 and parts[-1].isdigit():
        signer_parts = parts[len(label_parts):-1]
        if signer_parts:
            return normalize_id("_".join(signer_parts))
    return "unknown"


def load_label_whitelist(path: Path) -> set[str]:
    labels = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        labels.add(line.upper())
    return labels


def normalize_pose(pose: PoseDict) -> PoseDict:
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
        scale = float(np.linalg.norm(np.asarray(shoulder_left) - np.asarray(shoulder_right)))
        if scale < 1e-4:
            scale = 0.25

    out: PoseDict = {}
    for key in JOINT_KEYS:
        pt = pose.get(key)
        if pt is None:
            continue
        out[key] = ((pt[0] - torso[0]) / scale, (pt[1] - torso[1]) / scale)
    return out


def pose_to_feature_vector(pose: PoseDict) -> np.ndarray:
    normalized = normalize_pose(pose)
    feat: List[float] = []
    for key in JOINT_KEYS:
        pt = normalized.get(key)
        if pt is None:
            feat.extend([0.0, 0.0])
        else:
            feat.extend([float(pt[0]), float(pt[1])])
    x = np.asarray(feat, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -20.0, 20.0)


def sample_frame_indices(frame_count: int, seq_len: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros((seq_len,), dtype=np.int32)
    if frame_count == 1:
        return np.zeros((seq_len,), dtype=np.int32)
    return np.linspace(0, frame_count - 1, seq_len).round().astype(np.int32)


def load_clip_sequence(path: Path, seq_len: int) -> np.ndarray:
    clip = json.loads(path.read_text(encoding="utf-8"))
    frames = clip.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in clip: {path}")
    idx = sample_frame_indices(len(frames), seq_len)
    seq = [pose_to_feature_vector(frames[i]) for i in idx]
    return np.stack(seq, axis=0)


def _sample_from_path(path: Path, split: Optional[str] = None) -> ClipSample:
    clip_name = path.stem
    label = infer_label_from_clip_name(clip_name)
    clip = json.loads(path.read_text(encoding="utf-8"))
    metadata = clip.get("metadata", {}) if isinstance(clip, dict) else {}
    signer_id = normalize_id(
        str(
            metadata.get("signer_id")
            or metadata.get("member")
            or metadata.get("signer")
            or infer_signer_from_clip_name(clip_name, label)
        )
    )
    session_id = normalize_id(
        str(metadata.get("session_id") or metadata.get("session") or clip_name)
    )
    source = normalize_id(str(metadata.get("source") or metadata.get("dataset") or "unknown"))
    frames = clip.get("frames", []) if isinstance(clip, dict) else []
    return ClipSample(
        path=path,
        clip_name=clip_name,
        label=label,
        signer_id=signer_id,
        session_id=session_id,
        source=source,
        split=split.lower() if split else None,
        num_frames=len(frames) if isinstance(frames, list) else 0,
    )


def _resolve_sample_path(entry: dict, clips_dir: Path) -> Path:
    raw_path = str(entry.get("path", "")).strip()
    clip_name = str(entry.get("clip", "")).strip()
    if raw_path:
        path = Path(raw_path)
        if not path.is_absolute():
            path = clips_dir / raw_path
        return path
    if clip_name:
        if clip_name.endswith(".json"):
            return clips_dir / clip_name
        return clips_dir / f"{clip_name}.json"
    raise ValueError("Manifest sample is missing both 'path' and 'clip'")


def load_clip_samples(
    clips_dir: Path,
    labels_file: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    split_filter: Optional[str] = None,
) -> List[ClipSample]:
    whitelist = load_label_whitelist(labels_file) if labels_file else None
    samples: List[ClipSample] = []

    if manifest_path is None:
        entries: Iterable[Path] = sorted(clips_dir.glob("*.json"))
        for path in entries:
            sample = _sample_from_path(path)
            if whitelist is not None and sample.label not in whitelist:
                continue
            samples.append(sample)
    else:
        content = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(content, dict):
            raw_samples = content.get("samples")
            if raw_samples is None:
                raw_samples = content.get("saved")
            if raw_samples is None:
                raw_samples = content
        else:
            raw_samples = content
        if not isinstance(raw_samples, list):
            raise ValueError("Manifest must be a list or a dict with a 'samples' list")
        for entry in raw_samples:
            if not isinstance(entry, dict):
                continue
            path = _resolve_sample_path(entry, clips_dir)
            if not path.exists():
                continue
            sample = _sample_from_path(path, split=entry.get("split"))
            if "label" in entry and str(entry["label"]).strip():
                sample.label = str(entry["label"]).strip().upper()
            elif "sign" in entry and str(entry["sign"]).strip():
                sample.label = str(entry["sign"]).strip().upper()
            if "signer_id" in entry and str(entry["signer_id"]).strip():
                sample.signer_id = normalize_id(str(entry["signer_id"]))
            elif "member" in entry and str(entry["member"]).strip():
                sample.signer_id = normalize_id(str(entry["member"]))
            if "session_id" in entry and str(entry["session_id"]).strip():
                sample.session_id = normalize_id(str(entry["session_id"]))
            if "source" in entry and str(entry["source"]).strip():
                sample.source = normalize_id(str(entry["source"]))
            if "split" in entry and str(entry["split"]).strip():
                sample.split = str(entry["split"]).strip().lower()
            if whitelist is not None and sample.label not in whitelist:
                continue
            samples.append(sample)

    if split_filter:
        split_filter = split_filter.lower()
        samples = [sample for sample in samples if sample.split == split_filter]
    return samples


def split_samples(
    samples: List[ClipSample],
    val_ratio: float,
    seed: int,
    group_by_signer: bool = True,
) -> Tuple[List[int], List[int]]:
    explicit_train = [idx for idx, sample in enumerate(samples) if sample.split == "train"]
    explicit_val = [idx for idx, sample in enumerate(samples) if sample.split == "val"]
    if explicit_train or explicit_val:
        assigned = set(explicit_train) | set(explicit_val)
        remaining = [idx for idx in range(len(samples)) if idx not in assigned]
        if not remaining:
            return sorted(explicit_train), sorted(explicit_val)
        remaining_samples = [samples[idx] for idx in remaining]
        extra_train, extra_val = split_samples(
            samples=remaining_samples,
            val_ratio=val_ratio,
            seed=seed,
            group_by_signer=group_by_signer,
        )
        explicit_train.extend(remaining[idx] for idx in extra_train)
        explicit_val.extend(remaining[idx] for idx in extra_val)
        return sorted(explicit_train), sorted(explicit_val)

    rng = np.random.default_rng(seed)
    by_label: Dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        by_label.setdefault(sample.label, []).append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for label, indices in by_label.items():
        if len(indices) == 1:
            train_idx.extend(indices)
            continue

        if group_by_signer:
            grouped: Dict[str, List[int]] = {}
            for idx in indices:
                grouped.setdefault(samples[idx].group_key, []).append(idx)
            group_keys = list(grouped.keys())
            rng.shuffle(group_keys)
            if len(group_keys) >= 2:
                target_groups = int(round(len(group_keys) * val_ratio))
                if len(group_keys) >= 3 and target_groups == 0 and val_ratio > 0:
                    target_groups = 1
                target_groups = max(0, min(target_groups, len(group_keys) - 1))
                chosen_val = set(group_keys[:target_groups])
                if not chosen_val and val_ratio > 0 and len(group_keys) >= 3:
                    chosen_val = {group_keys[0]}
                for group_key, group_indices in grouped.items():
                    if group_key in chosen_val:
                        val_idx.extend(group_indices)
                    else:
                        train_idx.extend(group_indices)
                continue

        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_val = int(round(len(shuffled) * val_ratio))
        if len(shuffled) >= 3 and n_val == 0 and val_ratio > 0:
            n_val = 1
        n_val = max(0, min(n_val, len(shuffled) - 1))
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    return sorted(train_idx), sorted(val_idx)
