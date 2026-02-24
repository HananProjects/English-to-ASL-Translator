import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "data" / "eval" / "sign_quality_report.json"
DEFAULT_OUT = REPO_ROOT / "data" / "labels_recommended.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate recommended ASL training labels from sign quality report."
    )
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--min-clips",
        type=int,
        default=3,
        help="Minimum clips required per label.",
    )
    p.add_argument(
        "--min-quality",
        type=float,
        default=0.82,
        help="Minimum quality_score_top3_avg required.",
    )
    p.add_argument(
        "--max-labels",
        type=int,
        default=30,
        help="Maximum number of labels to keep (0 = no cap).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.report.read_text(encoding="utf-8"))
    signs = data.get("top_signs", [])

    filtered = []
    for row in signs:
        label = str(row.get("label", "")).upper()
        n = int(row.get("num_clips", 0))
        q = float(row.get("quality_score_top3_avg", 0.0))
        if not label:
            continue
        if n < args.min_clips:
            continue
        if q < args.min_quality:
            continue
        filtered.append((label, n, q))

    filtered.sort(key=lambda x: (-x[2], -x[1], x[0]))
    if args.max_labels > 0:
        filtered = filtered[: args.max_labels]

    lines = [
        "# Auto-generated recommended ASL training labels",
        f"# source={args.report}",
        f"# min_clips={args.min_clips} min_quality={args.min_quality} max_labels={args.max_labels}",
    ]
    lines.extend(label for label, _, _ in filtered)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"recommended_labels={len(filtered)}")
    print(f"out={args.out}")
    if filtered:
        preview = ", ".join(f"{l}({q:.2f},{n})" for l, n, q in filtered[:15])
        print("top:", preview)


if __name__ == "__main__":
    main()
