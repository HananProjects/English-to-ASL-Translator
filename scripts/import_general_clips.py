import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def parse_args():
    p = argparse.ArgumentParser(
        description="Import only general ASL signs from a local video repo/folder."
    )
    p.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Local path to downloaded videos (e.g. cloned SignLanguage repo).",
    )
    p.add_argument(
        "--labels-file",
        type=Path,
        default=Path("data/labels_general.txt"),
        help="Uppercase labels to keep (one per line).",
    )
    p.add_argument("--out-dir", type=Path, default=Path("ui/animation/clips"))
    p.add_argument("--report", type=Path, default=Path("data/general_import_report.json"))
    p.add_argument("--target-fps", type=int, default=12)
    p.add_argument("--max-per-label", type=int, default=2)
    p.add_argument("--min-frames", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=84)
    p.add_argument("--min-pose-ratio", type=float, default=0.6)
    p.add_argument("--min-hand-ratio", type=float, default=0.15)
    p.add_argument("--limit", type=int, default=0, help="Max saved clips total (0 = no cap).")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def normalize_key(text: str) -> str:
    out = text.strip().lower()
    out = re.sub(r"[^a-z0-9]+", "_", out)
    out = re.sub(r"_+", "_", out).strip("_")
    return out


def load_labels(path: Path) -> List[str]:
    labels: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        labels.append(line.upper())
    return sorted(set(labels))


def iter_videos(root: Path):
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            yield p


def infer_label_candidates(video_path: Path, source_root: Path) -> List[str]:
    rel = video_path.relative_to(source_root)
    candidates = []
    candidates.append(normalize_key(video_path.stem))
    for part in rel.parts[:-1]:
        candidates.append(normalize_key(part))
    out = [c for c in candidates if c]
    dedup: List[str] = []
    seen = set()
    for c in out:
        if c not in seen:
            dedup.append(c)
            seen.add(c)
    return dedup


def pick_label(video_path: Path, source_root: Path, allowed: set[str]) -> Optional[str]:
    candidates = infer_label_candidates(video_path, source_root)
    for c in candidates:
        token = c.upper()
        if token in allowed:
            return token

    for c in candidates:
        token = c.upper()
        for allowed_token in allowed:
            if token.startswith(allowed_token + "_") or token.endswith("_" + allowed_token):
                return allowed_token
    return None


def next_variant_index(out_dir: Path, label: str) -> int:
    prefix = normalize_key(label)
    pat = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?\.json$")
    highest = 0
    for p in out_dir.glob(f"{prefix}*.json"):
        m = pat.match(p.name)
        if not m:
            continue
        idx = int(m.group(1)) if m.group(1) else 1
        highest = max(highest, idx)
    return highest


def remove_existing_variants(out_dir: Path, label: str):
    prefix = normalize_key(label)
    pat = re.compile(rf"^{re.escape(prefix)}(?:_(\d+))?\.json$")
    for p in out_dir.glob(f"{prefix}*.json"):
        if pat.match(p.name):
            p.unlink(missing_ok=True)


def extract_clip(video_path: Path, holistic, target_fps: int, cv2, mediapipe_to_pose_dict):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-3:
        src_fps = 30.0

    sample_period = 1.0 / max(1, target_fps)
    next_sample_t = 0.0
    frame_idx = 0
    sampled = 0
    pose_frames = 0
    hand_frames = 0
    out_frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / src_fps
        frame_idx += 1
        if t + 1e-9 < next_sample_t:
            continue
        next_sample_t += sample_period
        sampled += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        pose = result.pose_landmarks.landmark if result.pose_landmarks else None
        left = result.left_hand_landmarks.landmark if result.left_hand_landmarks else None
        right = result.right_hand_landmarks.landmark if result.right_hand_landmarks else None
        if not pose:
            continue

        pose_frames += 1
        if left or right:
            hand_frames += 1
        out_frames.append(mediapipe_to_pose_dict(pose, left, right))

    cap.release()
    return {
        "fps": target_fps,
        "frames": out_frames,
        "stats": {
            "sampled_frames": sampled,
            "pose_frames": pose_frames,
            "hand_frames": hand_frames,
            "pose_ratio": (pose_frames / sampled) if sampled else 0.0,
            "hand_ratio": (hand_frames / pose_frames) if pose_frames else 0.0,
        },
    }


def main():
    args = parse_args()
    try:
        import cv2
        import mediapipe as mp
        from core.vision.pose_adapter import mediapipe_to_pose_dict
    except Exception as e:
        raise RuntimeError(
            "Missing dependencies for import. Install with: pip install opencv-contrib-python mediapipe"
        ) from e

    if not args.source_root.exists():
        raise FileNotFoundError(f"Source root not found: {args.source_root}")
    if not args.labels_file.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_file}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    labels = load_labels(args.labels_file)
    allowed = set(labels)

    if args.overwrite:
        for label in labels:
            remove_existing_variants(args.out_dir, label)

    per_label_count: Dict[str, int] = {
        label: next_variant_index(args.out_dir, label) for label in labels
    }

    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    report = {"saved": [], "skipped": []}
    saved_total = 0
    videos = list(iter_videos(args.source_root))

    for i, video in enumerate(videos, start=1):
        if args.limit and saved_total >= args.limit:
            break

        label = pick_label(video, args.source_root, allowed)
        if not label:
            continue
        if per_label_count[label] >= args.max_per_label:
            continue

        print(f"[import-general] {i}/{len(videos)} {video}")
        clip = extract_clip(video, mp_holistic, args.target_fps, cv2, mediapipe_to_pose_dict)
        if clip is None:
            report["skipped"].append({"source": str(video), "reason": "video_open_failed"})
            continue

        frame_count = len(clip["frames"])
        pose_ratio = clip["stats"]["pose_ratio"]
        hand_ratio = clip["stats"]["hand_ratio"]

        if frame_count < args.min_frames:
            report["skipped"].append(
                {"source": str(video), "label": label, "reason": "too_short", "frame_count": frame_count}
            )
            continue
        if frame_count > args.max_frames:
            report["skipped"].append(
                {"source": str(video), "label": label, "reason": "too_long", "frame_count": frame_count}
            )
            continue
        if pose_ratio < args.min_pose_ratio or hand_ratio < args.min_hand_ratio:
            report["skipped"].append(
                {
                    "source": str(video),
                    "label": label,
                    "reason": "quality_gate_failed",
                    "frame_count": frame_count,
                    "pose_ratio": pose_ratio,
                    "hand_ratio": hand_ratio,
                }
            )
            continue

        per_label_count[label] += 1
        idx = per_label_count[label]
        clip_name = f"{normalize_key(label)}_{idx}"
        out_path = args.out_dir / f"{clip_name}.json"
        out_path.write_text(
            json.dumps({"fps": clip["fps"], "frames": clip["frames"]}, indent=2),
            encoding="utf-8",
        )
        saved_total += 1
        report["saved"].append(
            {
                "source": str(video),
                "label": label,
                "clip": clip_name,
                "frame_count": frame_count,
                "pose_ratio": pose_ratio,
                "hand_ratio": hand_ratio,
            }
        )

    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved clips: {len(report['saved'])}")
    print(f"Skipped videos: {len(report['skipped'])}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
