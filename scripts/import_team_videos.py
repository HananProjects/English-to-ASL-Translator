import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp

# Allow running as: python -m scripts.import_team_videos
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.vision.pose_adapter import mediapipe_to_pose_dict

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Batch-import teammate ASL videos into clip JSON files using "
            "MediaPipe landmark extraction."
        )
    )
    parser.add_argument(
        "--videos-root",
        default="data/team_videos",
        help=(
            "Input root expected as data/team_videos/<member>/<sign>/*.mp4 "
            "(also supports flat sign folders)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="ui/animation/clips",
        help="Directory where output clip JSON files are written.",
    )
    parser.add_argument(
        "--report",
        default="data/team_import_report.json",
        help="Path to write import report JSON.",
    )
    parser.add_argument("--target-fps", type=int, default=12)
    parser.add_argument("--min-frames", type=int, default=12)
    parser.add_argument("--min-pose-ratio", type=float, default=0.6)
    parser.add_argument("--min-hand-ratio", type=float, default=0.15)
    parser.add_argument(
        "--members",
        default="",
        help="Optional comma-separated member filter.",
    )
    parser.add_argument(
        "--signs",
        default="",
        help="Optional comma-separated sign filter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated clip names.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of clips to save (0 = no limit).",
    )
    return parser.parse_args()


def normalize_key(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def list_videos(root: Path) -> List[Path]:
    out: List[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            out.append(path)
    return sorted(out)


def infer_member_and_sign(path: Path, root: Path) -> Tuple[str, str]:
    rel = path.relative_to(root)
    parts = rel.parts
    if len(parts) >= 3:
        member = normalize_key(parts[0])
        sign = normalize_key(parts[1])
        return member or "unknown", sign or "unknown"
    if len(parts) >= 2:
        member = "unknown"
        sign = normalize_key(parts[0])
        return member, sign or "unknown"
    return "unknown", "unknown"


def existing_index(out_dir: Path, sign: str, member: str) -> int:
    pat = re.compile(rf"^{re.escape(sign)}_{re.escape(member)}_(\d+)\.json$")
    highest = 0
    for p in out_dir.glob(f"{sign}_{member}_*.json"):
        m = pat.match(p.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest


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

    members_filter = {normalize_key(x) for x in args.members.split(",") if x.strip()}
    signs_filter = {normalize_key(x) for x in args.signs.split(",") if x.strip()}
    keep_all_members = len(members_filter) == 0
    keep_all_signs = len(signs_filter) == 0

    videos = list_videos(videos_root)
    if not videos:
        raise RuntimeError(f"No video files found under: {videos_root}")

    per_key_index: Dict[Tuple[str, str], int] = {}
    report = {"saved": [], "skipped": []}

    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    saved = 0
    for video in videos:
        member, sign = infer_member_and_sign(video, videos_root)
        if sign == "unknown":
            report["skipped"].append(
                {"source": str(video), "reason": "missing_sign_folder"}
            )
            continue
        if not keep_all_members and member not in members_filter:
            continue
        if not keep_all_signs and sign not in signs_filter:
            continue
        if args.limit and saved >= args.limit:
            break

        clip = extract_clip_from_video(
            video_path=video,
            holistic=mp_holistic,
            target_fps=args.target_fps,
        )
        if clip is None:
            report["skipped"].append({"source": str(video), "reason": "video_open_failed"})
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
                    "source": str(video),
                    "member": member,
                    "sign": sign,
                    "reason": "quality_gate_failed",
                    "frame_count": frame_count,
                    "pose_ratio": pose_ratio,
                    "hand_ratio": hand_ratio,
                }
            )
            continue

        key = (sign, member)
        if key not in per_key_index:
            per_key_index[key] = existing_index(out_dir, sign, member)
        per_key_index[key] += 1
        idx = per_key_index[key]
        clip_name = f"{sign}_{member}_{idx:02d}"
        out_path = out_dir / f"{clip_name}.json"

        if out_path.exists() and not args.overwrite:
            report["skipped"].append(
                {
                    "source": str(video),
                    "member": member,
                    "sign": sign,
                    "reason": "output_exists",
                    "output": str(out_path),
                }
            )
            per_key_index[key] -= 1
            continue

        out_data = {"fps": clip["fps"], "frames": clip["frames"]}
        out_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        saved += 1

        report["saved"].append(
            {
                "source": str(video),
                "member": member,
                "sign": sign,
                "clip": clip_name,
                "output": str(out_path),
                "frame_count": frame_count,
                "pose_ratio": pose_ratio,
                "hand_ratio": hand_ratio,
            }
        )

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved clips: {len(report['saved'])}")
    print(f"Skipped videos: {len(report['skipped'])}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
