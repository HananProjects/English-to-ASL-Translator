import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = REPO_ROOT / "ui" / "animation" / "clips"
DEFAULT_OUT = REPO_ROOT / "models" / "asl_landmark_classifier_v2.npz"

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
    p = argparse.ArgumentParser(
        description="Train ASL sequence classifier from clip JSON landmarks."
    )
    p.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-samples-per-label",
        type=int,
        default=2,
        help="Drop labels with fewer than this many clip samples.",
    )
    p.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional newline-separated uppercase label whitelist.",
    )
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


def load_clip_sequence(path: Path, seq_len: int) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        clip = json.load(f)
    frames = clip.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"No frames in clip: {path}")
    idx = sample_frame_indices(len(frames), seq_len)
    seq = [pose_to_feature_vector(frames[i]) for i in idx]
    x = np.stack(seq, axis=0)
    return x


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


def load_label_whitelist(path: Path) -> set[str]:
    labels = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        labels.add(line.upper())
    return labels


def build_dataset(
    clips_dir: Path,
    seq_len: int,
    labels_file: Path | None,
    min_samples_per_label: int,
):
    whitelist = load_label_whitelist(labels_file) if labels_file else None

    X_list: List[np.ndarray] = []
    y_tokens: List[str] = []
    counts: Dict[str, int] = {}
    files = sorted(clips_dir.glob("*.json"))
    for fp in files:
        token = infer_label_from_clip_name(fp.stem)
        if whitelist is not None and token not in whitelist:
            continue
        try:
            seq = load_clip_sequence(fp, seq_len=seq_len)
        except Exception:
            continue
        X_list.append(seq)
        y_tokens.append(token)
        counts[token] = counts.get(token, 0) + 1

    if not X_list:
        raise RuntimeError("No training samples found from clips.")

    keep_labels = {
        token for token, c in counts.items()
        if c >= max(1, int(min_samples_per_label))
    }
    dropped = sorted(
        (token, counts[token]) for token in counts
        if token not in keep_labels
    )

    X_keep: List[np.ndarray] = []
    y_keep_tokens: List[str] = []
    for x, token in zip(X_list, y_tokens):
        if token in keep_labels:
            X_keep.append(x)
            y_keep_tokens.append(token)

    if not X_keep:
        raise RuntimeError("No samples left after min-samples-per-label filter.")

    labels = sorted(set(y_keep_tokens))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y = np.asarray([label_to_idx[t] for t in y_keep_tokens], dtype=np.int64)
    X = np.stack(X_keep, axis=0).astype(np.float32)
    kept_counts = {lab: counts[lab] for lab in labels}
    return X, y, labels, kept_counts, dropped


def train_val_split(X: np.ndarray, y: np.ndarray, val_split: float, seed: int):
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(max(1, round(n * val_split))) if n >= 5 else 1
    n_val = min(n_val, n - 1) if n > 1 else 0
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]
    return X[tr_idx], y[tr_idx], X[val_idx], y[val_idx]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
    logits = np.clip(logits, -60.0, 60.0)
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-9)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.argmax(logits, axis=1)
    return float((pred == y).mean())


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    X, y, labels, kept_counts, dropped = build_dataset(
        args.clips_dir,
        args.seq_len,
        args.labels_file,
        args.min_samples_per_label,
    )
    n, seq_len, feat_dim = X.shape
    n_classes = len(labels)
    flat_dim = seq_len * feat_dim

    X_flat = X.reshape(n, flat_dim)
    mean = X_flat.mean(axis=0, keepdims=True)
    std = X_flat.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    Xn = (X_flat - mean) / std
    Xn = np.clip(Xn, -8.0, 8.0).astype(np.float32)
    Xn = np.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)

    Xtr, ytr, Xva, yva = train_val_split(Xn, y, args.val_split, args.seed)
    Ytr = one_hot(ytr, n_classes)

    rng = np.random.default_rng(args.seed)
    W = (rng.normal(0, 0.02, size=(flat_dim, n_classes))).astype(np.float32)
    b = np.zeros((n_classes,), dtype=np.float32)

    for epoch in range(1, args.epochs + 1):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = Xtr @ W + b
        logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
        logits = np.clip(logits, -60.0, 60.0)
        probs = softmax(logits)
        grad_logits = (probs - Ytr) / max(1, Xtr.shape[0])
        grad_logits = np.nan_to_num(grad_logits, nan=0.0, posinf=0.0, neginf=0.0)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            dW = Xtr.T @ grad_logits + args.l2 * W
        db = grad_logits.sum(axis=0)
        dW = np.nan_to_num(dW, nan=0.0, posinf=0.0, neginf=0.0)
        db = np.nan_to_num(db, nan=0.0, posinf=0.0, neginf=0.0)
        W -= args.lr * dW.astype(np.float32)
        b -= args.lr * db.astype(np.float32)
        W = np.clip(np.nan_to_num(W, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        b = np.clip(np.nan_to_num(b, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)

        if epoch % 25 == 0 or epoch == 1 or epoch == args.epochs:
            tr_acc = accuracy(logits, ytr)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                va_logits = Xva @ W + b if Xva.shape[0] > 0 else logits
            va_logits = np.nan_to_num(va_logits, nan=0.0, posinf=60.0, neginf=-60.0)
            va_logits = np.clip(va_logits, -60.0, 60.0)
            va_acc = accuracy(va_logits, yva) if Xva.shape[0] > 0 else tr_acc
            print(
                f"[epoch {epoch:03d}] train_acc={tr_acc:.3f} "
                f"val_acc={va_acc:.3f} samples={n} classes={n_classes}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        W=W.astype(np.float32),
        b=b.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        labels=np.asarray(labels),
        seq_len=np.asarray(seq_len, dtype=np.int32),
        feature_dim=np.asarray(feat_dim, dtype=np.int32),
    )
    print(f"Saved model: {args.out}")
    print(f"labels={len(labels)} seq_len={seq_len} feature_dim={feat_dim}")
    print(f"kept_labels={len(labels)} min_samples_per_label={args.min_samples_per_label}")
    if dropped:
        print("dropped_labels:", ", ".join(f"{t}({c})" for t, c in dropped))
    top_counts = sorted(kept_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    print("top_label_counts:", ", ".join(f"{t}({c})" for t, c in top_counts))


if __name__ == "__main__":
    main()
