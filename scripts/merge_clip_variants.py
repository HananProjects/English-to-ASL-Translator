import argparse
import json
import math
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_DIR = REPO_ROOT / "ui" / "animation" / "clips"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple sign clip variants into one averaged clip."
    )
    parser.add_argument(
        "--clips",
        nargs="+",
        required=True,
        help="Input clip names without .json, e.g. go_1 go_2 go_3",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output clip name without .json. Defaults to inferred '<base>_avg'.",
    )
    parser.add_argument(
        "--clip-dir",
        default=str(DEFAULT_CLIP_DIR),
        help="Directory containing clip JSON files.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Target FPS for merged clip.",
    )
    parser.add_argument(
        "--duration-mode",
        choices=("median", "mean", "min", "max"),
        default="median",
        help="How to choose merged duration from input clips.",
    )
    return parser.parse_args()


def infer_output_name(input_names: list[str]) -> str:
    if not input_names:
        return "merged_clip"
    first = input_names[0]
    if "_" in first:
        return f"{first.rsplit('_', 1)[0]}_avg"
    return f"{first}_avg"


def load_clip(clip_dir: Path, clip_name: str) -> dict:
    clip_path = clip_dir / f"{clip_name}.json"
    if not clip_path.exists():
        raise FileNotFoundError(f"Clip not found: {clip_path}")
    with open(clip_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    fps = float(data.get("fps", 0))
    frames = data.get("frames", [])
    if fps <= 0 or not isinstance(frames, list) or not frames:
        raise ValueError(f"Clip has invalid fps/frames: {clip_path}")
    return {"name": clip_name, "fps": fps, "frames": frames}


def interpolate_pose(a: dict, b: dict, t: float) -> dict:
    out = {}
    joints = set(a.keys()) | set(b.keys())
    for joint in joints:
        av = a.get(joint)
        bv = b.get(joint)
        if av is None:
            out[joint] = (float(bv[0]), float(bv[1]))
            continue
        if bv is None:
            out[joint] = (float(av[0]), float(av[1]))
            continue
        out[joint] = (
            float(av[0]) + (float(bv[0]) - float(av[0])) * t,
            float(av[1]) + (float(bv[1]) - float(av[1])) * t,
        )
    return out


def sample_pose(clip: dict, t_seconds: float) -> dict:
    frames = clip["frames"]
    fps = clip["fps"]
    frame_pos = max(0.0, t_seconds * fps)
    base = int(math.floor(frame_pos))
    if base >= len(frames) - 1:
        frame = frames[-1]
        return {k: (float(v[0]), float(v[1])) for k, v in frame.items()}
    frac = frame_pos - base
    return interpolate_pose(frames[base], frames[base + 1], frac)


def clip_duration_seconds(clip: dict) -> float:
    return len(clip["frames"]) / clip["fps"]


def choose_duration(durations: list[float], mode: str) -> float:
    if mode == "mean":
        return sum(durations) / len(durations)
    if mode == "min":
        return min(durations)
    if mode == "max":
        return max(durations)
    return median(durations)


def merge_pose_samples(samples: list[dict]) -> dict:
    merged = {}
    joints = set()
    for pose in samples:
        joints.update(pose.keys())
    for joint in joints:
        xs = []
        ys = []
        for pose in samples:
            point = pose.get(joint)
            if point is None:
                continue
            xs.append(float(point[0]))
            ys.append(float(point[1]))
        if xs:
            merged[joint] = [sum(xs) / len(xs), sum(ys) / len(ys)]
    return merged


def main():
    args = parse_args()
    clip_dir = Path(args.clip_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)

    clips = [load_clip(clip_dir, name) for name in args.clips]
    durations = [clip_duration_seconds(c) for c in clips]
    merged_duration = choose_duration(durations, args.duration_mode)
    target_fps = max(1, int(args.fps))
    target_frames = max(1, int(round(merged_duration * target_fps)))

    merged_frames = []
    for i in range(target_frames):
        t = i / target_fps
        samples = [sample_pose(c, t) for c in clips]
        merged_frames.append(merge_pose_samples(samples))

    out_name = args.output.strip() or infer_output_name(args.clips)
    out_path = clip_dir / f"{out_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"fps": target_fps, "frames": merged_frames}, f, indent=2)

    print(f"Merged {len(clips)} clips -> {out_path}")
    print(f"Duration: {merged_duration:.3f}s, FPS: {target_fps}, Frames: {target_frames}")


if __name__ == "__main__":
    main()
