import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = REPO_ROOT / "ui" / "animation" / "clips"
DEFAULT_OUT = REPO_ROOT / "models" / "asl_landmark_classifier_v2.npz"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.asl_to_english.dataset_utils import load_clip_samples, load_clip_sequence, split_samples
from core.asl_to_english.temporal_model import (
    normalize_input,
    save_linear_model,
    save_temporal_conv_model,
    softmax,
    temporal_conv_backward,
    temporal_conv_forward,
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
    p.add_argument(
        "--model-kind",
        choices=("temporal_cnn", "linear"),
        default="temporal_cnn",
        help="Sequence model architecture to train.",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=48,
        help="Hidden channel count for temporal_cnn.",
    )
    p.add_argument(
        "--kernel-size",
        type=int,
        default=3,
        help="Temporal convolution kernel size for temporal_cnn.",
    )
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
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest/metadata JSON with sample-level split and signer metadata.",
    )
    p.add_argument(
        "--group-by-signer",
        action="store_true",
        help="Keep same-signer samples together during train/val splitting when possible.",
    )
    return p.parse_args()


def build_dataset(
    clips_dir: Path,
    seq_len: int,
    labels_file: Path | None,
    min_samples_per_label: int,
    manifest_path: Path | None,
):
    X_list: List[np.ndarray] = []
    y_tokens: List[str] = []
    counts: Dict[str, int] = {}
    sample_records = []
    for sample in load_clip_samples(
        clips_dir=clips_dir,
        labels_file=labels_file,
        manifest_path=manifest_path,
    ):
        try:
            seq = load_clip_sequence(sample.path, seq_len=seq_len)
        except Exception:
            continue
        X_list.append(seq)
        y_tokens.append(sample.label)
        sample_records.append(sample)
        counts[sample.label] = counts.get(sample.label, 0) + 1

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
    sample_keep = []
    for x, token, sample in zip(X_list, y_tokens, sample_records):
        if token in keep_labels:
            X_keep.append(x)
            y_keep_tokens.append(token)
            sample_keep.append(sample)

    if not X_keep:
        raise RuntimeError("No samples left after min-samples-per-label filter.")

    labels = sorted(set(y_keep_tokens))
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    y = np.asarray([label_to_idx[t] for t in y_keep_tokens], dtype=np.int64)
    X = np.stack(X_keep, axis=0).astype(np.float32)
    kept_counts = {lab: counts[lab] for lab in labels}
    return X, y, labels, kept_counts, dropped, sample_keep


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
    pred = np.argmax(logits, axis=1)
    return float((pred == y).mean())


def build_split_manifest(
    samples,
    train_idx: List[int],
    val_idx: List[int],
    labels: List[str],
    kept_counts: Dict[str, int],
    dropped,
    args: argparse.Namespace,
):
    split_by_index = {idx: "train" for idx in train_idx}
    split_by_index.update({idx: "val" for idx in val_idx})
    manifest_samples = []
    for idx, sample in enumerate(samples):
        manifest_samples.append(
            {
                "clip": sample.clip_name,
                "path": str(sample.path),
                "label": sample.label,
                "signer_id": sample.signer_id,
                "session_id": sample.session_id,
                "source": sample.source,
                "split": split_by_index.get(idx, sample.split or "train"),
            }
        )
    return {
        "clips_dir": str(args.clips_dir),
        "labels_file": str(args.labels_file) if args.labels_file else None,
        "source_manifest": str(args.manifest) if args.manifest else None,
        "seq_len": args.seq_len,
        "labels": labels,
        "counts": kept_counts,
        "dropped_labels": [{"label": label, "count": count} for label, count in dropped],
        "samples": manifest_samples,
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    X, y, labels, kept_counts, dropped, samples = build_dataset(
        args.clips_dir,
        args.seq_len,
        args.labels_file,
        args.min_samples_per_label,
        args.manifest,
    )
    n, seq_len, feat_dim = X.shape
    n_classes = len(labels)
    flat_dim = seq_len * feat_dim

    X_flat = X.reshape(n, flat_dim)
    mean = X_flat.mean(axis=0, keepdims=True)
    std = X_flat.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    Xn = normalize_input(X, mean, std)

    train_idx, val_idx = split_samples(
        samples=samples,
        val_ratio=args.val_split,
        seed=args.seed,
        group_by_signer=args.group_by_signer,
    )
    Xtr = Xn[train_idx]
    ytr = y[train_idx]
    Xva = Xn[val_idx] if val_idx else np.zeros((0, Xn.shape[1], Xn.shape[2]), dtype=np.float32)
    yva = y[val_idx] if val_idx else np.zeros((0,), dtype=np.int64)
    Ytr = one_hot(ytr, n_classes)

    print(
        f"[split] train={len(train_idx)} val={len(val_idx)} "
        f"group_by_signer={'yes' if args.group_by_signer else 'no'}"
    )

    rng = np.random.default_rng(args.seed)
    linear_W = None
    linear_b = None
    conv_W = None
    conv_b = None
    fc_W = None
    fc_b = None
    if args.model_kind == "linear":
        linear_W = (rng.normal(0, 0.02, size=(flat_dim, n_classes))).astype(np.float32)
        linear_b = np.zeros((n_classes,), dtype=np.float32)
    else:
        kernel_size = max(1, int(args.kernel_size))
        hidden_dim = max(8, int(args.hidden_dim))
        conv_W = rng.normal(0, 0.02, size=(hidden_dim, kernel_size, feat_dim)).astype(np.float32)
        conv_b = np.zeros((hidden_dim,), dtype=np.float32)
        fc_W = rng.normal(0, 0.02, size=(hidden_dim, n_classes)).astype(np.float32)
        fc_b = np.zeros((n_classes,), dtype=np.float32)

    for epoch in range(1, args.epochs + 1):
        if args.model_kind == "linear":
            Xtr_flat = Xtr.reshape(Xtr.shape[0], -1)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                logits = Xtr_flat @ linear_W + linear_b
            logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
            logits = np.clip(logits, -60.0, 60.0)
        else:
            logits, cache = temporal_conv_forward(Xtr, conv_W, conv_b, fc_W, fc_b)
            logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
            logits = np.clip(logits, -60.0, 60.0)
        probs = softmax(logits)
        grad_logits = (probs - Ytr) / max(1, Xtr.shape[0])
        grad_logits = np.nan_to_num(grad_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if args.model_kind == "linear":
            Xtr_flat = Xtr.reshape(Xtr.shape[0], -1)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                dW = Xtr_flat.T @ grad_logits + args.l2 * linear_W
            db = grad_logits.sum(axis=0)
            dW = np.nan_to_num(dW, nan=0.0, posinf=0.0, neginf=0.0)
            db = np.nan_to_num(db, nan=0.0, posinf=0.0, neginf=0.0)
            linear_W -= args.lr * dW.astype(np.float32)
            linear_b -= args.lr * db.astype(np.float32)
            linear_W = np.clip(np.nan_to_num(linear_W, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
            linear_b = np.clip(np.nan_to_num(linear_b, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
        else:
            dW1, db1, dW2, db2 = temporal_conv_backward(grad_logits, cache, conv_W, fc_W)
            dW1 += args.l2 * conv_W
            dW2 += args.l2 * fc_W
            conv_W -= args.lr * np.nan_to_num(dW1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            conv_b -= args.lr * np.nan_to_num(db1, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            fc_W -= args.lr * np.nan_to_num(dW2, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            fc_b -= args.lr * np.nan_to_num(db2, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            conv_W = np.clip(np.nan_to_num(conv_W, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
            conv_b = np.clip(np.nan_to_num(conv_b, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
            fc_W = np.clip(np.nan_to_num(fc_W, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
            fc_b = np.clip(np.nan_to_num(fc_b, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)

        if epoch % 25 == 0 or epoch == 1 or epoch == args.epochs:
            tr_acc = accuracy(logits, ytr)
            if Xva.shape[0] > 0:
                if args.model_kind == "linear":
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                        va_logits = Xva.reshape(Xva.shape[0], -1) @ linear_W + linear_b
                else:
                    va_logits, _ = temporal_conv_forward(Xva, conv_W, conv_b, fc_W, fc_b)
            else:
                va_logits = logits
            va_logits = np.nan_to_num(va_logits, nan=0.0, posinf=60.0, neginf=-60.0)
            va_logits = np.clip(va_logits, -60.0, 60.0)
            va_acc = accuracy(va_logits, yva) if Xva.shape[0] > 0 else tr_acc
            print(
                f"[epoch {epoch:03d}] train_acc={tr_acc:.3f} "
                f"val_acc={va_acc:.3f} samples={n} classes={n_classes}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.model_kind == "linear":
        save_linear_model(
            args.out,
            W=linear_W,
            b=linear_b,
            mean=mean,
            std=std,
            labels=labels,
            seq_len=seq_len,
            feature_dim=feat_dim,
        )
    else:
        save_temporal_conv_model(
            args.out,
            W1=conv_W,
            b1=conv_b,
            W2=fc_W,
            b2=fc_b,
            mean=mean,
            std=std,
            labels=labels,
            seq_len=seq_len,
            feature_dim=feat_dim,
        )
    split_manifest = build_split_manifest(
        samples=samples,
        train_idx=train_idx,
        val_idx=val_idx,
        labels=labels,
        kept_counts=kept_counts,
        dropped=dropped,
        args=args,
    )
    split_manifest_path = args.out.with_suffix(".manifest.json")
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    print(f"Saved model: {args.out}")
    print(f"Saved split manifest: {split_manifest_path}")
    print(f"model_kind={args.model_kind} labels={len(labels)} seq_len={seq_len} feature_dim={feat_dim}")
    print(f"kept_labels={len(labels)} min_samples_per_label={args.min_samples_per_label}")
    if dropped:
        print("dropped_labels:", ", ".join(f"{t}({c})" for t, c in dropped))
    top_counts = sorted(kept_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
    print("top_label_counts:", ", ".join(f"{t}({c})" for t, c in top_counts))


if __name__ == "__main__":
    main()
