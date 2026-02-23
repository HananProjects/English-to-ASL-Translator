"""Check which ASL labels can be played by the current dictionary + clip files.

Usage:
  .\\.venv\\Scripts\\python.exe scripts/check_sign_playability.py
  .\\.venv\\Scripts\\python.exe scripts/check_sign_playability.py --labels-file data/labels_merged_branch.txt
  .\\.venv\\Scripts\\python.exe scripts/check_sign_playability.py --report data/sign_playability_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.english_to_asl.dictionary.asl_signs import ASL_SIGNS


CLIPS_DIR = Path("ui/animation/clips")
DEFAULT_LABELS_FILE = Path("data/labels_merged_branch.txt")


def _load_labels(path: Path) -> list[str]:
    labels: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        labels.append(line.upper())
    return labels


def _resolve_clip_names(entry: dict) -> list[str]:
    if "clip" in entry:
        return [str(entry["clip"])]
    if "clips" in entry and isinstance(entry["clips"], list):
        return [str(x) for x in entry["clips"]]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=DEFAULT_LABELS_FILE,
        help=f"Path to label list (default: {DEFAULT_LABELS_FILE})",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional JSON report output path",
    )
    args = parser.parse_args()

    labels = _load_labels(args.labels_file)
    clip_stems = {p.stem for p in CLIPS_DIR.glob("*.json")}

    playable: list[dict] = []
    unplayable: list[dict] = []

    for label in labels:
        entry = ASL_SIGNS.get(label)
        if not entry:
            unplayable.append(
                {"label": label, "reason": "missing_in_asl_signs", "candidates": []}
            )
            continue

        candidates = _resolve_clip_names(entry)
        existing = [c for c in candidates if c in clip_stems]

        if existing:
            playable.append(
                {
                    "label": label,
                    "configured": candidates,
                    "existing": existing,
                }
            )
        else:
            unplayable.append(
                {
                    "label": label,
                    "reason": "configured_clips_missing",
                    "candidates": candidates,
                }
            )

    total = len(labels)
    playable_count = len(playable)
    unplayable_count = len(unplayable)

    print(f"labels_checked: {total}")
    print(f"playable: {playable_count}")
    print(f"unplayable: {unplayable_count}")

    if unplayable:
        print("\nUnplayable labels:")
        for row in unplayable:
            cands = ", ".join(row["candidates"]) if row["candidates"] else "-"
            print(f"  {row['label']}: {row['reason']} (configured: {cands})")

    if args.report:
        report = {
            "labels_checked": total,
            "playable_count": playable_count,
            "unplayable_count": unplayable_count,
            "playable": playable,
            "unplayable": unplayable,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nReport written: {args.report}")


if __name__ == "__main__":
    main()
