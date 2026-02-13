import argparse
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import cv2
import mediapipe as mp

# Allow running as: python scripts/import_wlasl_clips.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.vision.pose_adapter import mediapipe_to_pose_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import usable WLASL videos into clip JSON files."
    )
    parser.add_argument(
        "--metadata",
        default="data/WLASL/start_kit/WLASL_v0.3.json",
        help="Path to WLASL_v0.3.json",
    )
    parser.add_argument(
        "--videos-dir",
        default="data/WLASL/start_kit/raw_videos",
        help="Directory containing downloaded WLASL videos.",
    )
    parser.add_argument(
        "--out-dir",
        default="ui/animation/clips",
        help="Directory where output clip JSON files are written.",
    )
    parser.add_argument("--target-fps", type=int, default=12, help="Output clip FPS.")
    parser.add_argument(
        "--max-per-gloss",
        type=int,
        default=3,
        help="Max number of clips to keep per gloss.",
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
        "--glosses",
        default="",
        help="Optional comma-separated gloss filter (e.g. hello,how,you).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of saved clips across all glosses (0 = no limit).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing clips for selected glosses.",
    )
    parser.add_argument(
        "--report",
        default="data/wlasl_import_report.json",
        help="Path to write JSON import report.",
    )
    return parser.parse_args()


def normalize_gloss(gloss: str) -> str:
    value = gloss.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def youtube_id_from_url(url: str) -> str | None:
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        if "youtu.be" in host:
            return parsed.path.strip("/") or None
        if "youtube.com" in host:
            qs = parse_qs(parsed.query)
            vals = qs.get("v", [])
            if vals:
                return vals[0]
    except Exception:
        return None
    return None


def resolve_video_path(videos_dir: Path, instance: dict) -> Path | None:
    video_id = str(instance.get("video_id", "")).strip()
    url = str(instance.get("url", "")).strip()
    candidates = []

    for ext in (".mp4", ".mkv", ".webm"):
        candidates.append(videos_dir / f"{video_id}{ext}")

    yid = youtube_id_from_url(url)
    if yid:
        for ext in (".mp4", ".mkv", ".webm"):
            candidates.append(videos_dir / f"{yid}{ext}")

    for path in candidates:
        if path.exists():
            return path
    return None


def existing_variant_count(out_dir: Path, gloss_key: str) -> int:
    pattern = re.compile(rf"^{re.escape(gloss_key)}(?:_(\d+))?\.json$")
    highest = 0
    for p in out_dir.glob(f"{gloss_key}*.json"):
        m = pattern.match(p.name)
        if not m:
            continue
        idx = int(m.group(1)) if m.group(1) else 1
        highest = max(highest, idx)
    return highest


def remove_existing_variants(out_dir: Path, gloss_key: str):
    pattern = re.compile(rf"^{re.escape(gloss_key)}(?:_(\d+))?\.json$")
    for p in out_dir.glob(f"{gloss_key}*.json"):
        if pattern.match(p.name):
            p.unlink(missing_ok=True)


def extract_clip_from_video(
    video_path: Path,
    holistic,
    target_fps: int,
):
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
    metadata_path = Path(args.metadata)
    videos_dir = Path(args.videos_dir)
    out_dir = Path(args.out_dir)
    report_path = Path(args.report)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos dir not found: {videos_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    gloss_filter = {
        normalize_gloss(x) for x in args.glosses.split(",") if x.strip()
    }
    keep_all = len(gloss_filter) == 0

    content = json.loads(metadata_path.read_text(encoding="utf-8"))

    if args.overwrite:
        targets = set()
        for entry in content:
            g = normalize_gloss(entry.get("gloss", ""))
            if keep_all or g in gloss_filter:
                targets.add(g)
        for g in targets:
            remove_existing_variants(out_dir, g)

    per_gloss_count = {}
    for entry in content:
        g = normalize_gloss(entry.get("gloss", ""))
        if keep_all or g in gloss_filter:
            per_gloss_count[g] = existing_variant_count(out_dir, g)

    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    report = {
        "saved": [],
        "skipped": [],
    }

    saved_total = 0
    for entry in content:
        raw_gloss = str(entry.get("gloss", "")).strip()
        gloss_key = normalize_gloss(raw_gloss)
        if not gloss_key:
            continue
        if not keep_all and gloss_key not in gloss_filter:
            continue

        if per_gloss_count.get(gloss_key, 0) >= args.max_per_gloss:
            continue

        instances = entry.get("instances", [])
        for inst in instances:
            if per_gloss_count.get(gloss_key, 0) >= args.max_per_gloss:
                break
            if args.limit and saved_total >= args.limit:
                break

            video_path = resolve_video_path(videos_dir, inst)
            if video_path is None:
                report["skipped"].append({
                    "gloss": raw_gloss,
                    "video_id": inst.get("video_id"),
                    "reason": "missing_video_file",
                })
                continue

            clip = extract_clip_from_video(
                video_path=video_path,
                holistic=mp_holistic,
                target_fps=args.target_fps,
            )
            if clip is None:
                report["skipped"].append({
                    "gloss": raw_gloss,
                    "video_id": inst.get("video_id"),
                    "reason": "video_open_failed",
                })
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
                report["skipped"].append({
                    "gloss": raw_gloss,
                    "video_id": inst.get("video_id"),
                    "reason": "quality_gate_failed",
                    "frame_count": frame_count,
                    "pose_ratio": pose_ratio,
                    "hand_ratio": hand_ratio,
                })
                continue

            per_gloss_count[gloss_key] = per_gloss_count.get(gloss_key, 0) + 1
            variant_idx = per_gloss_count[gloss_key]
            clip_name = f"{gloss_key}_{variant_idx}"
            out_path = out_dir / f"{clip_name}.json"

            out_data = {
                "fps": clip["fps"],
                "frames": clip["frames"],
            }
            out_path.write_text(
                json.dumps(out_data, indent=2),
                encoding="utf-8",
            )
            saved_total += 1

            report["saved"].append({
                "gloss": raw_gloss,
                "clip": clip_name,
                "video_id": inst.get("video_id"),
                "source": str(video_path),
                "frame_count": frame_count,
                "pose_ratio": pose_ratio,
                "hand_ratio": hand_ratio,
            })

        if args.limit and saved_total >= args.limit:
            break

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved clips: {len(report['saved'])}")
    print(f"Skipped videos: {len(report['skipped'])}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
