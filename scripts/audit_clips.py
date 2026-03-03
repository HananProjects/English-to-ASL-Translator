import argparse
import json
import math
from pathlib import Path


def wrist_motion_score(frames):
    keys = ("hand_left", "hand_right")
    total = 0.0
    steps = 0
    for key in keys:
        prev = None
        for frame in frames:
            v = frame.get(key) if isinstance(frame, dict) else None
            if isinstance(v, list) and len(v) >= 2:
                cur = (float(v[0]), float(v[1]))
                if prev is not None:
                    total += math.dist(cur, prev)
                    steps += 1
                prev = cur
    return (total / steps) if steps else 0.0


def main():
    parser = argparse.ArgumentParser(description="Audit ASL clip JSON quality")
    parser.add_argument(
        "--clips-dir",
        default="ui/animation/clips",
        help="Directory containing clip JSON files",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=5.0,
        help="Flag clips longer than this duration",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.4,
        help="Flag clips shorter than this duration",
    )
    parser.add_argument(
        "--min-motion",
        type=float,
        default=0.004,
        help="Flag clips with very low mean wrist motion per frame step",
    )
    parser.add_argument(
        "--report-json",
        default="data/eval/clip_audit_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()

    clips_dir = Path(args.clips_dir)
    files = sorted(clips_dir.glob("*.json"))
    issues = []
    ok = 0

    for path in files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            fps = float(data.get("fps", 0))
            frames = data.get("frames", [])
            if fps <= 0 or not isinstance(frames, list) or not frames:
                issues.append(
                    {
                        "clip": path.name,
                        "reason": "invalid_fps_or_frames",
                        "fps": fps,
                        "frame_count": len(frames) if isinstance(frames, list) else 0,
                    }
                )
                continue

            duration = len(frames) / fps
            motion = wrist_motion_score(frames)
            reasons = []
            if duration > args.max_seconds:
                reasons.append("too_long")
            if duration < args.min_seconds:
                reasons.append("too_short")
            if motion < args.min_motion:
                reasons.append("low_motion")

            if reasons:
                issues.append(
                    {
                        "clip": path.name,
                        "reason": ",".join(reasons),
                        "fps": fps,
                        "frame_count": len(frames),
                        "duration": round(duration, 3),
                        "wrist_motion_score": round(motion, 6),
                    }
                )
            else:
                ok += 1
        except Exception as e:
            issues.append({"clip": path.name, "reason": "parse_error", "error": str(e)})

    report = {
        "clips_dir": str(clips_dir),
        "total_clips": len(files),
        "ok_clips": ok,
        "issue_count": len(issues),
        "thresholds": {
            "max_seconds": args.max_seconds,
            "min_seconds": args.min_seconds,
            "min_motion": args.min_motion,
        },
        "issues": issues,
    }

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Total clips: {len(files)}")
    print(f"OK clips: {ok}")
    print(f"Issues: {len(issues)}")
    print(f"Report: {report_path}")
    if issues:
        print("Top suspicious clips:")
        for row in sorted(
            issues, key=lambda x: float(x.get("duration", 0.0)), reverse=True
        )[:20]:
            dur = row.get("duration", 0.0)
            print(f"  {row['clip']}: {row['reason']} (duration={dur}s)")


if __name__ == "__main__":
    main()

