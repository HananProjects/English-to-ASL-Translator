import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = REPO_ROOT / "ui" / "animation" / "clips"
DEFAULT_MODEL = REPO_ROOT / "models" / "asl_landmark_classifier_v2.npz"
DEFAULT_REPORT = REPO_ROOT / "data" / "eval" / "asl_to_english_eval.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.asl_to_english.dataset_utils import load_clip_samples, load_clip_sequence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ASL sequence model on clip JSON set.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest/metadata JSON with sample-level split and signer metadata.",
    )
    p.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default=None,
        help="If manifest samples include split labels, evaluate only this split.",
    )
    p.add_argument(
        "--labels-file",
        type=Path,
        default=None,
        help="Optional newline-separated uppercase label whitelist.",
    )
    return p.parse_args()


def load_model(path: Path):
    model = np.load(path, allow_pickle=True)
    W = model["W"].astype(np.float32)
    b = model["b"].astype(np.float32)
    mean = model["mean"].astype(np.float32)
    std = model["std"].astype(np.float32)
    W = np.clip(np.nan_to_num(W, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    b = np.clip(np.nan_to_num(b, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    mean = np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
    std[std < 1e-6] = 1.0
    std = np.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
    labels = [str(x).upper() for x in model["labels"].tolist()]
    seq_len = int(model["seq_len"])
    feat_dim = int(model["feature_dim"])
    return W, b, mean, std, labels, seq_len, feat_dim


def top_k_indices(probs: np.ndarray, k: int) -> List[int]:
    if probs.ndim != 1 or probs.size == 0:
        return []
    k = max(1, min(k, probs.size))
    indices = np.argsort(probs)[::-1][:k]
    return [int(idx) for idx in indices]


def main() -> None:
    args = parse_args()
    W, b, mean, std, labels, seq_len, feat_dim = load_model(args.model)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    manifest_path = args.manifest
    if manifest_path is None:
        candidate_manifest = args.model.with_suffix(".manifest.json")
        if candidate_manifest.exists():
            manifest_path = candidate_manifest

    split_filter = None
    evaluated_split = "all"
    if args.split:
        split_filter = None if args.split == "all" else args.split
        evaluated_split = args.split
    elif manifest_path is not None:
        split_filter = "val"
        evaluated_split = "val"

    samples = load_clip_samples(
        clips_dir=args.clips_dir,
        labels_file=args.labels_file,
        manifest_path=manifest_path,
        split_filter=split_filter,
    )

    rows = []
    per_label_totals: Dict[str, int] = {}
    per_label_correct: Dict[str, int] = {}
    per_signer_totals: Dict[str, int] = {}
    per_signer_correct: Dict[str, int] = {}
    top3_correct = 0

    for sample in samples:
        try:
            seq = load_clip_sequence(sample.path, seq_len=seq_len)
        except Exception:
            continue
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
        prob_row = probs[0]

        pred_idx = int(np.argmax(prob_row))
        pred_label = labels[pred_idx]
        conf = float(prob_row[pred_idx])
        top3 = top_k_indices(prob_row, 3)
        top3_labels = [labels[idx] for idx in top3]
        correct = pred_label == sample.label
        top3_hit = sample.label in top3_labels

        per_label_totals[sample.label] = per_label_totals.get(sample.label, 0) + 1
        per_label_correct[sample.label] = per_label_correct.get(sample.label, 0) + int(correct)
        per_signer_totals[sample.signer_id] = per_signer_totals.get(sample.signer_id, 0) + 1
        per_signer_correct[sample.signer_id] = per_signer_correct.get(sample.signer_id, 0) + int(correct)
        top3_correct += int(top3_hit)

        rows.append(
            {
                "clip": sample.path.name,
                "true_label": sample.label,
                "pred_label": pred_label,
                "confidence": conf,
                "correct": correct,
                "top3_labels": top3_labels,
                "top3_hit": top3_hit,
                "signer_id": sample.signer_id,
                "session_id": sample.session_id,
                "source": sample.source,
                "split": sample.split,
            }
        )

    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    acc = (correct / total) if total else 0.0
    top3_acc = (top3_correct / total) if total else 0.0

    confusion: Dict[str, Dict[str, int]] = {}
    for row in rows:
        confusion.setdefault(row["true_label"], {})
        confusion[row["true_label"]][row["pred_label"]] = (
            confusion[row["true_label"]].get(row["pred_label"], 0) + 1
        )

    per_label_accuracy = {
        label: {
            "total": per_label_totals[label],
            "correct": per_label_correct.get(label, 0),
            "accuracy": (
                per_label_correct.get(label, 0) / per_label_totals[label]
                if per_label_totals[label]
                else 0.0
            ),
        }
        for label in sorted(per_label_totals)
    }
    per_signer_accuracy = {
        signer: {
            "total": per_signer_totals[signer],
            "correct": per_signer_correct.get(signer, 0),
            "accuracy": (
                per_signer_correct.get(signer, 0) / per_signer_totals[signer]
                if per_signer_totals[signer]
                else 0.0
            ),
        }
        for signer in sorted(per_signer_totals)
    }

    unknown_true = sorted(
        {row["true_label"] for row in rows if row["true_label"] not in label_to_idx}
    )
    top_errors = [row for row in rows if not row["correct"]]
    top_errors = sorted(top_errors, key=lambda row: row["confidence"], reverse=True)[:30]

    report = {
        "model": str(args.model),
        "clips_dir": str(args.clips_dir),
        "manifest": str(manifest_path) if manifest_path else None,
        "evaluated_split": evaluated_split,
        "total_samples": total,
        "correct": correct,
        "accuracy": acc,
        "top3_accuracy": top3_acc,
        "label_count": len(labels),
        "unknown_true_labels_not_in_model": unknown_true,
        "top_errors": top_errors,
        "confusion": confusion,
        "per_label_accuracy": per_label_accuracy,
        "per_signer_accuracy": per_signer_accuracy,
    }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"total={total} correct={correct} acc={acc:.3f} top3_acc={top3_acc:.3f} "
        f"split={evaluated_split}"
    )
    if unknown_true:
        print("unknown_true_labels_not_in_model:", ", ".join(unknown_true))
    print(f"report={args.report}")


if __name__ == "__main__":
    main()
