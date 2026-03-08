import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DATA_DIR = REPO_ROOT / "data"
EVAL_DIR = DATA_DIR / "eval"
MODELS_DIR = REPO_ROOT / "models"

DEFAULT_QUALITY_REPORT = EVAL_DIR / "sign_quality_report_full.json"
DEFAULT_RECOMMENDED_LABELS = DATA_DIR / "labels_recommended_verify.txt"
DEFAULT_NEW_LABELS = DATA_DIR / "labels_new_for_training.txt"
DEFAULT_EXPANDED_LABELS = DATA_DIR / "labels_expanded_for_training.txt"
DEFAULT_SUMMARY_REPORT = EVAL_DIR / "training_expansion_report.json"
DEFAULT_MODEL_IN = MODELS_DIR / "asl_landmark_classifier_v2.npz"
DEFAULT_MODEL_OUT = MODELS_DIR / "asl_landmark_classifier_v3.npz"
DEFAULT_EVAL_REPORT = EVAL_DIR / "asl_to_english_eval_v3.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run ASL gesture verification and prepare label expansion for training. "
            "Optionally retrain and evaluate in one workflow."
        )
    )
    p.add_argument("--clips-dir", type=Path, default=REPO_ROOT / "ui" / "animation" / "clips")

    p.add_argument("--quality-report", type=Path, default=DEFAULT_QUALITY_REPORT)
    p.add_argument("--recommended-labels-out", type=Path, default=DEFAULT_RECOMMENDED_LABELS)
    p.add_argument("--new-labels-out", type=Path, default=DEFAULT_NEW_LABELS)
    p.add_argument("--expanded-labels-out", type=Path, default=DEFAULT_EXPANDED_LABELS)
    p.add_argument("--summary-report", type=Path, default=DEFAULT_SUMMARY_REPORT)

    p.add_argument("--min-clips", type=int, default=3)
    p.add_argument("--min-quality", type=float, default=0.82)
    p.add_argument("--max-labels", type=int, default=40)
    p.add_argument("--top", type=int, default=60, help="Top/bottom entries to keep in quality report.")

    p.add_argument("--current-model", type=Path, default=DEFAULT_MODEL_IN)

    p.add_argument("--train", action="store_true", help="Train a new model using expanded labels.")
    p.add_argument("--eval", action="store_true", help="Evaluate model after training.")
    p.add_argument("--train-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--eval-report", type=Path, default=DEFAULT_EVAL_REPORT)
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--l2", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-samples-per-label", type=int, default=2)

    p.add_argument("--dry-run", action="store_true", help="Prepare outputs without running retrain/eval.")
    return p.parse_args()


def run_cmd(cmd: List[str]) -> None:
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


def read_labels_file(path: Path) -> List[str]:
    labels: List[str] = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        labels.append(line.upper())
    return labels


def read_model_labels(path: Path) -> List[str]:
    if not path.exists():
        return []
    model = np.load(path, allow_pickle=True)
    raw = model["labels"].tolist()
    return [str(x).upper() for x in raw]


def write_labels(path: Path, header: List[str], labels: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = list(header)
    lines.extend(labels)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize_eval_report(path: Path, focus_labels: set[str]) -> dict:
    if not path.exists():
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    rows = report.get("top_errors", [])
    total = int(report.get("total_samples", 0))
    correct = int(report.get("correct", 0))
    overall_acc = float(report.get("accuracy", 0.0))

    # eval_asl_sequence_model does not emit all rows; build focused metrics from confusion.
    confusion = report.get("confusion", {})
    focused_total = 0
    focused_correct = 0
    for true_label, pred_map in confusion.items():
        t = str(true_label).upper()
        if t not in focus_labels:
            continue
        for pred_label, count in pred_map.items():
            c = int(count)
            focused_total += c
            if str(pred_label).upper() == t:
                focused_correct += c

    focused_acc = (focused_correct / focused_total) if focused_total else 0.0
    unknown_true = report.get("unknown_true_labels_not_in_model", [])

    return {
        "overall_total": total,
        "overall_correct": correct,
        "overall_accuracy": overall_acc,
        "focused_total": focused_total,
        "focused_correct": focused_correct,
        "focused_accuracy": focused_acc,
        "unknown_true_label_count": len(unknown_true),
        "unknown_true_labels_preview": [str(x) for x in unknown_true[:30]],
        "top_errors_preview_count": len(rows),
    }


def main() -> None:
    args = parse_args()

    args.quality_report.parent.mkdir(parents=True, exist_ok=True)
    args.summary_report.parent.mkdir(parents=True, exist_ok=True)

    rank_script = SCRIPTS_DIR / "rank_sign_quality.py"
    recommend_script = SCRIPTS_DIR / "recommend_training_labels.py"
    train_script = SCRIPTS_DIR / "train_asl_sequence_model.py"
    eval_script = SCRIPTS_DIR / "eval_asl_sequence_model.py"

    # 1) Verify clip/sign quality.
    run_cmd(
        [
            sys.executable,
            str(rank_script),
            "--clips-dir",
            str(args.clips_dir),
            "--report",
            str(args.quality_report),
            "--top",
            str(args.top),
        ]
    )

    # 2) Recommend high-quality labels.
    run_cmd(
        [
            sys.executable,
            str(recommend_script),
            "--report",
            str(args.quality_report),
            "--out",
            str(args.recommended_labels_out),
            "--min-clips",
            str(args.min_clips),
            "--min-quality",
            str(args.min_quality),
            "--max-labels",
            str(args.max_labels),
        ]
    )

    recommended = sorted(set(read_labels_file(args.recommended_labels_out)))
    current_model_labels = sorted(set(read_model_labels(args.current_model)))
    new_labels = sorted(set(recommended) - set(current_model_labels))
    expanded_labels = sorted(set(current_model_labels) | set(recommended))

    write_labels(
        args.new_labels_out,
        [
            "# Auto-generated labels not present in current model",
            f"# current_model={args.current_model}",
            f"# recommended_file={args.recommended_labels_out}",
        ],
        new_labels,
    )

    write_labels(
        args.expanded_labels_out,
        [
            "# Auto-generated expanded label set for retraining",
            f"# current_model={args.current_model}",
            f"# recommended_file={args.recommended_labels_out}",
        ],
        expanded_labels,
    )

    model_used_for_eval = args.current_model
    trained = False

    # 3) Optional retrain.
    if args.train and not args.dry_run:
        run_cmd(
            [
                sys.executable,
                str(train_script),
                "--clips-dir",
                str(args.clips_dir),
                "--out",
                str(args.train_out),
                "--seq-len",
                str(args.seq_len),
                "--epochs",
                str(args.epochs),
                "--lr",
                str(args.lr),
                "--l2",
                str(args.l2),
                "--val-split",
                str(args.val_split),
                "--seed",
                str(args.seed),
                "--min-samples-per-label",
                str(args.min_samples_per_label),
                "--labels-file",
                str(args.expanded_labels_out),
            ]
        )
        trained = True
        model_used_for_eval = args.train_out

    # 4) Optional evaluation.
    evaluated = False
    eval_summary = {}
    if args.eval and not args.dry_run:
        run_cmd(
            [
                sys.executable,
                str(eval_script),
                "--model",
                str(model_used_for_eval),
                "--clips-dir",
                str(args.clips_dir),
                "--report",
                str(args.eval_report),
            ]
        )
        evaluated = True
        eval_summary = summarize_eval_report(args.eval_report, set(expanded_labels))

    summary = {
        "clips_dir": str(args.clips_dir),
        "quality_report": str(args.quality_report),
        "recommended_labels_file": str(args.recommended_labels_out),
        "new_labels_file": str(args.new_labels_out),
        "expanded_labels_file": str(args.expanded_labels_out),
        "current_model": str(args.current_model),
        "train_enabled": bool(args.train),
        "eval_enabled": bool(args.eval),
        "dry_run": bool(args.dry_run),
        "trained": trained,
        "evaluated": evaluated,
        "train_out": str(args.train_out),
        "eval_report": str(args.eval_report),
        "recommended_label_count": len(recommended),
        "current_model_label_count": len(current_model_labels),
        "new_label_count": len(new_labels),
        "expanded_label_count": len(expanded_labels),
        "new_labels_preview": new_labels[:30],
        "eval_summary": eval_summary,
    }

    args.summary_report.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"recommended_labels={len(recommended)}")
    print(f"current_model_labels={len(current_model_labels)}")
    print(f"new_labels={len(new_labels)}")
    print(f"expanded_labels={len(expanded_labels)}")
    print(f"summary={args.summary_report}")


if __name__ == "__main__":
    main()
