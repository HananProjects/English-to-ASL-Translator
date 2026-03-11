import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a prioritized re-record target list from an ASL eval report."
    )
    parser.add_argument(
        "--eval-report",
        type=Path,
        required=True,
        help="Path to eval JSON report produced by eval_asl_sequence_model.py",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional manifest to attach per-label clip counts.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "data" / "eval" / "rerecord_targets.json",
        help="Where to write the prioritized target report JSON.",
    )
    parser.add_argument(
        "--out-labels",
        type=Path,
        default=REPO_ROOT / "data" / "labels_rerecord_priority.txt",
        help="Where to write the newline-separated priority label list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.eval_report.read_text(encoding="utf-8"))
    manifest = None
    if args.manifest and args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))

    manifest_counts = {}
    if manifest and isinstance(manifest, dict):
        for row in manifest.get("coverage", []):
            label = str(row.get("label", "")).strip().upper()
            if label:
                manifest_counts[label] = int(row.get("count", 0))

    top_errors = report.get("top_errors", [])
    per_label = report.get("per_label_accuracy", {})
    pair_counts = Counter()

    targets = []
    for label, stats in per_label.items():
        acc = float(stats.get("accuracy", 0.0))
        total = int(stats.get("total", 0))
        correct = int(stats.get("correct", 0))
        if total <= 0:
            continue
        if acc >= 1.0:
            continue
        errors = [row for row in top_errors if row.get("true_label") == label]
        top_confusion = errors[0]["pred_label"] if errors else None
        top_confidence = float(errors[0]["confidence"]) if errors else 0.0
        targets.append(
            {
                "label": label,
                "accuracy": acc,
                "total": total,
                "correct": correct,
                "current_clip_count": manifest_counts.get(label),
                "top_confusion": top_confusion,
                "top_confusion_confidence": top_confidence,
                "recommended_new_clips": 12 if acc == 0.0 else 8,
            }
        )

    for row in top_errors:
        true_label = str(row.get("true_label", "")).strip().upper()
        pred_label = str(row.get("pred_label", "")).strip().upper()
        if true_label and pred_label and true_label != pred_label:
            pair_counts[(true_label, pred_label)] += 1

    targets.sort(
        key=lambda row: (
            row["accuracy"],
            -(row["top_confusion_confidence"] or 0.0),
            row["label"],
        )
    )
    confusion_groups = [
        {"true_label": true_label, "pred_label": pred_label, "count": count}
        for (true_label, pred_label), count in pair_counts.most_common()
    ]

    payload = {
        "eval_report": str(args.eval_report),
        "manifest": str(args.manifest) if args.manifest else None,
        "target_count": len(targets),
        "targets": targets,
        "confusion_groups": confusion_groups,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_labels.parent.mkdir(parents=True, exist_ok=True)
    args.out_labels.write_text(
        "\n".join(row["label"] for row in targets) + ("\n" if targets else ""),
        encoding="utf-8",
    )

    print(
        f"[rerecord] targets={len(targets)} "
        f"out_json={args.out_json} out_labels={args.out_labels}"
    )
    if targets:
        preview = ", ".join(
            f"{row['label']}->{row['top_confusion']}" for row in targets[:10]
        )
        print(f"[rerecord] preview: {preview}")


if __name__ == "__main__":
    main()
