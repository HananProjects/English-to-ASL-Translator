import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp

# Allow running as: python scripts/import_team_videos.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.vision.pose_adapter import mediapipe_to_pose_dict

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import local teammate sign videos into clip JSON files."
    )
    parser.add_argument(
        "--videos-root",
        default="extra_clips",
        help="Root folder containing teammate videos.",
    )
    parser.add_argument(
        "--out-dir",
        default="ui/animation/clips",
        help="Directory where output clip JSON files are written.",
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=12,
        help="Output clip FPS.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=12,
        help="Minimum number of output frames required for a usable clip.",
    )
    parser.add_argument(
        "--min-pose-ratio",
        type=float,
        default=0.6,
        help="Minimum sampled-frame pose detection ratio [0..1].",
    )
    parser.add_argument(
        "--min-hand-ratio",
        type=float,
        default=0.15,
        help="Minimum pose-frame hand detection ratio [0..1].",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output clip name if present.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of videos to process (0 = no limit).",
    )
    parser.add_argument(
        "--report",
        default="data/team_import_report.json",
        help="Path to write JSON import report.",
    )
    return parser.parse_args()


def normalize_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def list_videos(root: Path):
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(p)
    return sorted(files)


def unique_out_name(base: str, out_dir: Path, overwrite: bool) -> str:
    if overwrite:
        return base
    candidate = base
    idx = 2
    while (out_dir / f"{candidate}.json").exists():
        candidate = f"{base}_{idx}"
        idx += 1
    return candidate


def extract_clip_from_video(video_path: Path, holistic, target_fps: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1e-3:
        src_fps = 30.0

    sample_period = 1.0 / max(1, target_fps)
    next_sample_t = 0.0
    frame_idx = 0

    sampled_frames = 0
    pose_frames = 0
    hand_frames = 0
    output_frames = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t = frame_idx / src_fps
        frame_idx += 1
        if t + 1e-9 < next_sample_t:
            continue
        next_sample_t += sample_period
        sampled_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        pose_landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
        left_hand_landmarks = result.left_hand_landmarks.landmark if result.left_hand_landmarks else None
        right_hand_landmarks = result.right_hand_landmarks.landmark if result.right_hand_landmarks else None

        if not pose_landmarks:
            continue

        pose_frames += 1
        if left_hand_landmarks or right_hand_landmarks:
            hand_frames += 1

        pose = mediapipe_to_pose_dict(
            pose_landmarks,
            left_hand_landmarks,
            right_hand_landmarks,
        )
        output_frames.append(pose)

    cap.release()

    pose_ratio = pose_frames / sampled_frames if sampled_frames else 0.0
    hand_ratio = hand_frames / pose_frames if pose_frames else 0.0

    return {
        "fps": target_fps,
        "frames": output_frames,
        "stats": {
            "sampled_frames": sampled_frames,
            "pose_frames": pose_frames,
            "hand_frames": hand_frames,
            "pose_ratio": pose_ratio,
            "hand_ratio": hand_ratio,
        },
    }


def main():
    args = parse_args()
    videos_root = Path(args.videos_root)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)

    if not videos_root.exists():
        raise FileNotFoundError(f"Videos root not found: {videos_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    videos = list_videos(videos_root)
    if args.limit > 0:
        videos = videos[: args.limit]

    print(f"[import] found videos: {len(videos)}")

    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    report = {"saved": [], "skipped": []}

    for idx, video_path in enumerate(videos, start=1):
        t0 = time.time()
        clip_base = normalize_name(video_path.stem)
        if not clip_base:
            report["skipped"].append(
                {"source": str(video_path), "reason": "invalid_filename"}
            )
            print(f"[skip] {idx}/{len(videos)} invalid_filename {video_path.name}")
            continue

        clip_name = unique_out_name(clip_base, out_dir, args.overwrite)
        out_path = out_dir / f"{clip_name}.json"
        print(f"[import] {idx}/{len(videos)} {video_path.name} -> {clip_name}.json")

        clip = extract_clip_from_video(
            video_path=video_path,
            holistic=mp_holistic,
            target_fps=args.target_fps,
        )
        if clip is None:
            report["skipped"].append(
                {"source": str(video_path), "reason": "video_open_failed"}
            )
            print("[skip] video_open_failed")
            continue

        frame_count = len(clip["frames"])
        pose_ratio = clip["stats"]["pose_ratio"]
        hand_ratio = clip["stats"]["hand_ratio"]
        usable = (
            frame_count >= args.min_frames
            and pose_ratio >= args.min_pose_ratio
            and hand_ratio >= args.min_hand_ratio
        )
        if not usable:
            report["skipped"].append(
                {
                    "source": str(video_path),
                    "reason": "quality_gate_failed",
                    "frame_count": frame_count,
                    "pose_ratio": pose_ratio,
                    "hand_ratio": hand_ratio,
                }
            )
            print(
                f"[skip] quality_gate_failed frames={frame_count} "
                f"pose_ratio={pose_ratio:.2f} hand_ratio={hand_ratio:.2f}"
            )
            continue

        out_data = {"fps": clip["fps"], "frames": clip["frames"]}
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")

        report["saved"].append(
            {
                "source": str(video_path),
                "clip": clip_name,
                "output": str(out_path),
                "frame_count": frame_count,
                "pose_ratio": pose_ratio,
                "hand_ratio": hand_ratio,
                "time_s": round(time.time() - t0, 2),
            }
        )
        print(f"[saved] clip={clip_name} frames={frame_count}")

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved clips: {len(report['saved'])}")
    print(f"Skipped videos: {len(report['skipped'])}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
