import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = REPO_ROOT / "ui" / "animation" / "clips"
DEFAULT_MODEL = REPO_ROOT / "models" / "asl_landmark_classifier_v2.npz"
DEFAULT_REPORT = REPO_ROOT / "data" / "eval" / "asl_to_english_eval.json"

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ASL sequence model on clip JSON set.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return p.parse_args()


def normalize_pose(pose: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
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

    out: Dict[str, Tuple[float, float]] = {}
    for key in JOINT_KEYS:
        pt = pose.get(key)
        if pt is None:
            continue
        out[key] = ((pt[0] - torso[0]) / scale, (pt[1] - torso[1]) / scale)
    return out


def pose_to_feature_vector(pose: Dict[str, Tuple[float, float]]) -> np.ndarray:
    n = normalize_pose(pose)
    feat: List[float] = []
    for key in JOINT_KEYS:
        pt = n.get(key)
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


def infer_label_from_clip_name(stem: str) -> str:
    parts = stem.lower().split("_")
    out = []
    for p in parts:
        if p.isdigit():
            break
        out.append(p)
    if not out:
        out = [parts[0]]
    return "_".join(out).upper()


def load_model(path: Path):
    m = np.load(path, allow_pickle=True)
    W = m["W"].astype(np.float32)
    b = m["b"].astype(np.float32)
    mean = m["mean"].astype(np.float32)
    std = m["std"].astype(np.float32)
    W = np.clip(np.nan_to_num(W, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    b = np.clip(np.nan_to_num(b, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std[std < 1e-6] = 1.0
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    labels = [str(x).upper() for x in m["labels"].tolist()]
    seq_len = int(m["seq_len"])
    feat_dim = int(m["feature_dim"])
    return W, b, mean, std, labels, seq_len, feat_dim


def main() -> None:
    args = parse_args()
    W, b, mean, std, labels, seq_len, feat_dim = load_model(args.model)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    rows = []
    for fp in sorted(args.clips_dir.glob("*.json")):
        true_label = infer_label_from_clip_name(fp.stem)
        with open(fp, "r", encoding="utf-8") as f:
            clip = json.load(f)
        frames = clip.get("frames", [])
        if not isinstance(frames, list) or not frames:
            continue
        idx = sample_frame_indices(len(frames), seq_len)
        seq = np.stack([pose_to_feature_vector(frames[i]) for i in idx], axis=0)
        if seq.shape[1] != feat_dim:
            continue
        x = seq.reshape(1, -1).astype(np.float32)
        x = (x - mean) / std
        x = np.clip(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = x @ W + b
        logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
        logits = np.clip(logits, -60.0, 60.0)
        logits = logits - logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs = probs / np.maximum(probs.sum(axis=1, keepdims=True), 1e-9)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        pred_idx = int(np.argmax(probs[0]))
        pred_label = labels[pred_idx]
        conf = float(probs[0, pred_idx])
        rows.append(
            {
                "clip": fp.name,
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": conf,
                "correct": pred_label == true_label,
            }
        )

    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    acc = (correct / total) if total else 0.0

    confusion: Dict[str, Dict[str, int]] = {}
    for r in rows:
        t = r["true_label"]
        p = r["pred_label"]
        confusion.setdefault(t, {})
        confusion[t][p] = confusion[t].get(p, 0) + 1

    unknown_true = sorted({r["true_label"] for r in rows if r["true_label"] not in label_to_idx})
    top_errors = [r for r in rows if not r["correct"]]
    top_errors = sorted(top_errors, key=lambda x: x["confidence"], reverse=True)[:30]

    report = {
        "model": str(args.model),
        "clips_dir": str(args.clips_dir),
        "total_samples": total,
        "correct": correct,
        "accuracy": acc,
        "unknown_true_labels_not_in_model": unknown_true,
        "top_errors": top_errors,
        "confusion": confusion,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"total={total} correct={correct} acc={acc:.3f}")
    if unknown_true:
        print("unknown_true_labels_not_in_model:", ", ".join(unknown_true))
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
