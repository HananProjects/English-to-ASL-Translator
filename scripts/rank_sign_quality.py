import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
CLIPS_DIR = REPO_ROOT / "ui" / "animation" / "clips"
DEFAULT_REPORT = REPO_ROOT / "data" / "eval" / "sign_quality_report.json"

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

HAND_KEYS: Tuple[str, ...] = (
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
    p = argparse.ArgumentParser(description="Rank clip/sign quality from landmark coverage.")
    p.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--top", type=int, default=30)
    return p.parse_args()


def infer_label(stem: str) -> str:
    parts = stem.lower().split("_")
    out = []
    for p in parts:
        if p.isdigit():
            break
        out.append(p)
    if not out:
        out = [parts[0]]
    return "_".join(out).upper()


def score_clip(frames: List[dict]) -> Tuple[float, float, float]:
    if not frames:
        return 0.0, 0.0, 0.0
    pose_cov = 0.0
    hand_cov = 0.0
    for f in frames:
        if not isinstance(f, dict):
            continue
        pose_cov += sum(1 for k in JOINT_KEYS if k in f) / float(len(JOINT_KEYS))
        hand_cov += sum(1 for k in HAND_KEYS if k in f) / float(len(HAND_KEYS))
    pose_cov /= float(len(frames))
    hand_cov /= float(len(frames))
    frame_factor = min(1.0, len(frames) / 60.0)
    # Weighted for recognition quality: tracking completeness matters most.
    score = 0.60 * pose_cov + 0.30 * hand_cov + 0.10 * frame_factor
    return score, pose_cov, hand_cov


def main() -> None:
    args = parse_args()
    clip_rows = []
    grouped: Dict[str, List[dict]] = {}

    for fp in sorted(args.clips_dir.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        frames = data.get("frames", [])
        if not isinstance(frames, list):
            continue
        label = infer_label(fp.stem)
        score, pose_cov, hand_cov = score_clip(frames)
        row = {
            "clip": fp.name,
            "label": label,
            "frames": len(frames),
            "score": round(score, 4),
            "pose_coverage": round(pose_cov, 4),
            "hand_coverage": round(hand_cov, 4),
        }
        clip_rows.append(row)
        grouped.setdefault(label, []).append(row)

    sign_rows = []
    for label, rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
        top3 = rows_sorted[:3]
        score = sum(r["score"] for r in top3) / max(1, len(top3))
        sign_rows.append(
            {
                "label": label,
                "num_clips": len(rows),
                "quality_score_top3_avg": round(score, 4),
                "best_clips": [r["clip"] for r in top3],
            }
        )

    sign_rows.sort(key=lambda r: r["quality_score_top3_avg"], reverse=True)
    clip_rows.sort(key=lambda r: r["score"], reverse=True)

    report = {
        "clips_dir": str(args.clips_dir),
        "total_clips_scored": len(clip_rows),
        "total_labels_scored": len(sign_rows),
        "top_signs": sign_rows[: args.top],
        "bottom_signs": list(reversed(sign_rows[-args.top:])),
        "top_clips": clip_rows[: args.top],
        "bottom_clips": list(reversed(clip_rows[-args.top:])),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"scored_clips={len(clip_rows)} scored_labels={len(sign_rows)}")
    print(f"report={args.report}")
    if sign_rows:
        top_preview = ", ".join(f"{r['label']}({r['quality_score_top3_avg']:.2f})" for r in sign_rows[:10])
        print("top_signs:", top_preview)


if __name__ == "__main__":
    main()
