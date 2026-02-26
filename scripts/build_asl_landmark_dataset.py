import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from core.english_to_asl.dictionary.asl_signs import ASL_SIGNS

# Keep the same joints used by the runtime recognizer.
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


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed-length ASL landmark sequence datasets from clip JSON files."
        )
    )
    parser.add_argument(
        "--labels",
        default="data/labels_100.txt",
        help="Path to canonical label file (one label per line).",
    )
    parser.add_argument(
        "--clips-dir",
        default="ui/animation/clips",
        help="Directory containing clip JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/datasets/asl_landmarks_v1",
        help="Where to write train/val/test NPZ files and metadata.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=48,
        help="Number of timesteps per sample sequence.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split shuffling.",
    )
    return parser.parse_args()


def load_labels(path: Path) -> List[str]:
    labels: List[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            labels.append(line.upper())
    deduped = sorted(set(labels))
    return deduped


def label_to_clips(label: str) -> List[str]:
    sign_def = ASL_SIGNS.get(label)
    if not sign_def:
        return []
    if isinstance(sign_def.get("clips"), list):
        return list(sign_def["clips"])
    clip = sign_def.get("clip")
    return [clip] if clip else []


def load_clip_frames(clip_path: Path) -> List[Dict[str, List[float]]]:
    with open(clip_path, "r", encoding="utf-8") as handle:
        clip = json.load(handle)
    frames = clip.get("frames", [])
    if not isinstance(frames, list):
        return []
    return frames


def normalize_frame(frame: Dict[str, List[float]]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, Tuple[float, float]] = {}

    torso = frame.get("torso")
    shoulder_left = frame.get("shoulder_left")
    shoulder_right = frame.get("shoulder_right")

    if torso is None:
        if shoulder_left and shoulder_right:
            torso = [
                (shoulder_left[0] + shoulder_right[0]) / 2.0,
                (shoulder_left[1] + shoulder_right[1]) / 2.0,
            ]
        elif shoulder_left:
            torso = shoulder_left
        elif shoulder_right:
            torso = shoulder_right
        else:
            torso = [0.5, 0.5]

    scale = 0.25
    if shoulder_left and shoulder_right:
        dx = shoulder_left[0] - shoulder_right[0]
        dy = shoulder_left[1] - shoulder_right[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 1e-4:
            scale = dist

    for key in JOINT_KEYS:
        point = frame.get(key)
        if not point:
            continue
        out[key] = (
            float(point[0] - torso[0]) / scale,
            float(point[1] - torso[1]) / scale,
        )

    return out


def sequence_from_frames(
    frames: List[Dict[str, List[float]]],
    seq_len: int,
) -> np.ndarray:
    if not frames:
        return np.zeros((seq_len, len(JOINT_KEYS) * 2), dtype=np.float32)

    normalized = [normalize_frame(frame) for frame in frames]
    if len(normalized) == 1:
        indices = np.zeros(seq_len, dtype=np.int64)
    else:
        positions = np.linspace(0, len(normalized) - 1, num=seq_len)
        indices = np.rint(positions).astype(np.int64)

    sequence = np.zeros((seq_len, len(JOINT_KEYS) * 2), dtype=np.float32)
    for t, idx in enumerate(indices):
        pose = normalized[int(idx)]
        feat: List[float] = []
        for key in JOINT_KEYS:
            point = pose.get(key)
            if point is None:
                feat.extend([0.0, 0.0])
            else:
                feat.extend([point[0], point[1]])
        sequence[t] = np.asarray(feat, dtype=np.float32)
    return sequence


def split_indices_per_label(
    label_sample_indices: Dict[int, List[int]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for _, indices in label_sample_indices.items():
        indices = list(indices)
        rng.shuffle(indices)

        n = len(indices)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)

        # Small-class safeguard: preserve train samples while still creating
        # meaningful eval splits when enough samples exist.
        if n >= 3 and n_val == 0:
            n_val = 1
        if n >= 5 and n_test == 0:
            n_test = 1

        if n - (n_test + n_val) < 1:
            overflow = (n_test + n_val) - (n - 1)
            while overflow > 0 and n_test > 0:
                n_test -= 1
                overflow -= 1
            while overflow > 0 and n_val > 0:
                n_val -= 1
                overflow -= 1

        test_slice = indices[:n_test]
        val_slice = indices[n_test:n_test + n_val]
        train_slice = indices[n_test + n_val:]

        test_idx.extend(test_slice)
        val_idx.extend(val_slice)
        train_idx.extend(train_slice)

    # Fallback for very small datasets: ensure a non-empty test split.
    if not test_idx and len(train_idx) >= 10:
        rng.shuffle(train_idx)
        holdout = max(1, int(round(len(train_idx) * test_ratio)))
        test_idx = train_idx[:holdout]
        train_idx = train_idx[holdout:]

    return train_idx, val_idx, test_idx


def save_split(
    name: str,
    indices: List[int],
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
):
    if not indices:
        split_X = np.zeros((0,) + X.shape[1:], dtype=np.float32)
        split_y = np.zeros((0,), dtype=np.int64)
    else:
        split_X = X[indices]
        split_y = y[indices]
    np.savez_compressed(output_dir / f"{name}.npz", X=split_X, y=split_y)


def main():
    args = parse_args()
    labels_path = Path(args.labels)
    clips_dir = Path(args.clips_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(labels_path)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    all_sequences: List[np.ndarray] = []
    all_targets: List[int] = []
    sample_meta: List[Dict[str, str]] = []
    missing_labels: List[str] = []
    missing_clips: List[str] = []

    for label in labels:
        clip_names = label_to_clips(label)
        if not clip_names:
            missing_labels.append(label)
            continue

        for clip_name in clip_names:
            clip_path = clips_dir / f"{clip_name}.json"
            if not clip_path.exists():
                missing_clips.append(f"{label}:{clip_name}")
                continue

            frames = load_clip_frames(clip_path)
            if not frames:
                continue

            sequence = sequence_from_frames(frames, seq_len=args.seq_len)
            all_sequences.append(sequence)
            all_targets.append(label_to_index[label])
            sample_meta.append(
                {
                    "label": label,
                    "clip": clip_name,
                    "path": str(clip_path),
                    "num_frames": len(frames),
                }
            )

    if not all_sequences:
        raise RuntimeError("No training samples found. Check labels and clips.")

    X = np.stack(all_sequences).astype(np.float32)
    y = np.asarray(all_targets, dtype=np.int64)

    label_sample_indices: Dict[int, List[int]] = {}
    for i, target in enumerate(y.tolist()):
        label_sample_indices.setdefault(target, []).append(i)

    train_idx, val_idx, test_idx = split_indices_per_label(
        label_sample_indices=label_sample_indices,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    save_split("train", train_idx, X, y, output_dir)
    save_split("val", val_idx, X, y, output_dir)
    save_split("test", test_idx, X, y, output_dir)

    counts = {
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
        "total": int(X.shape[0]),
    }
    per_label_counts = {
        label: len(label_sample_indices.get(label_to_index[label], []))
        for label in labels
    }

    metadata = {
        "labels_path": str(labels_path),
        "clips_dir": str(clips_dir),
        "seq_len": args.seq_len,
        "feature_dim": len(JOINT_KEYS) * 2,
        "labels": labels,
        "label_to_index": label_to_index,
        "counts": counts,
        "per_label_counts": per_label_counts,
        "missing_labels": missing_labels,
        "missing_clips": sorted(set(missing_clips)),
        "samples": sample_meta,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        "[dataset] built",
        f"total={counts['total']}",
        f"train={counts['train']}",
        f"val={counts['val']}",
        f"test={counts['test']}",
    )
    if missing_labels:
        print(f"[dataset] labels without ASL_SIGNS entries: {len(missing_labels)}")
    if missing_clips:
        print(f"[dataset] missing clip files referenced: {len(set(missing_clips))}")


if __name__ == "__main__":
    main()
