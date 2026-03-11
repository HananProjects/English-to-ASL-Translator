import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.asl_to_english.dataset_utils import (
    infer_label_from_clip_name,
    infer_signer_from_clip_name,
    load_label_whitelist,
)


CLIPS_DIR = REPO_ROOT / "ui" / "animation" / "clips"
DEFAULT_OUT = REPO_ROOT / "data" / "training_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a training manifest and coverage report from clip JSON files."
    )
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    parser.add_argument("--labels-file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--min-samples-per-label",
        type=int,
        default=2,
        help="Mark labels below this count as sparse in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    whitelist = load_label_whitelist(args.labels_file) if args.labels_file else None

    clips = sorted(args.clips_dir.glob("*.json"))
    samples = []
    counts = Counter()
    signer_counts = defaultdict(set)

    for clip_path in clips:
        clip_name = clip_path.stem
        label = infer_label_from_clip_name(clip_name)
        if whitelist is not None and label not in whitelist:
            continue
        signer_id = infer_signer_from_clip_name(clip_name, label)
        source = "team" if signer_id != "unknown" else "unknown"
        samples.append(
            {
                "clip": clip_name,
                "path": str(clip_path),
                "label": label,
                "signer_id": signer_id,
                "session_id": clip_name,
                "source": source,
            }
        )
        counts[label] += 1
        signer_counts[label].add(signer_id)

    sparse_labels = sorted(
        label for label, count in counts.items() if count < max(1, args.min_samples_per_label)
    )
    coverage = [
        {
            "label": label,
            "count": counts[label],
            "unique_signers": len({s for s in signer_counts[label] if s != "unknown"}),
        }
        for label in sorted(counts)
    ]

    report = {
        "clips_dir": str(args.clips_dir),
        "labels_file": str(args.labels_file) if args.labels_file else None,
        "total_clips": len(samples),
        "total_labels": len(counts),
        "min_samples_per_label": args.min_samples_per_label,
        "sparse_labels": sparse_labels,
        "coverage": coverage,
        "samples": samples,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"[manifest] clips={len(samples)} labels={len(counts)} "
        f"sparse_labels={len(sparse_labels)} out={args.out}"
    )
    top_counts = counts.most_common(20)
    if top_counts:
        print(
            "[manifest] top_counts:",
            ", ".join(f"{label}({count})" for label, count in top_counts),
        )


if __name__ == "__main__":
    main()
