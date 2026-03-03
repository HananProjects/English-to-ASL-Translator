# scripts/play_clip.py
import argparse
import importlib
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from core.sequencing.sign_sequencer import SignEvent
from ui.animation.poses.basic_poses import POSES


def resolve_animation_view_class():
    candidates = (
        "ui_polishses.widgets.animation_view",
        "ui.widgets.animation_view",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        view_cls = getattr(module, "ASLAnimationView", None)
        if view_cls is not None:
            return view_cls, module_name
    raise ImportError("Could not find ASLAnimationView in ui_polishses or ui modules.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", help="Path or name, e.g. ui/animation/clips/about_1.json or about_1.json")
    ap.add_argument("--all-good", action="store_true", help="Play all clips from ui/animation/clips")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of clips in --all-good mode (0 = no limit)")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle clip order in --all-good mode")
    ap.add_argument("--loop", action="store_true", help="Loop playback")
    ap.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    ap.add_argument(
        "--return-seconds",
        type=float,
        default=0.0,
        help="Seconds to lower hands into REST pose after a single clip",
    )
    ap.add_argument(
        "--tail-frames",
        type=int,
        default=0,
        help="Extra hold frames at the end to prevent early cutoff",
    )
    args = ap.parse_args()

    clips_dir = (Path(__file__).resolve().parents[1] / "ui" / "animation" / "clips").resolve()
    view_cls, view_module = resolve_animation_view_class()
    ui_playback_speed = max(0.01, float(getattr(view_cls, "PLAYBACK_SPEED", 1.0)))

    def read_clip_duration(path: Path) -> float:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        fps = float(data.get("fps", 12.0))
        frames = data.get("frames", [])
        if fps <= 0:
            raise ValueError(f"Invalid fps={fps} in {path}")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"No frames in {path}")
        # Match ASLAnimationView timing exactly: frame_pos uses PLAYBACK_SPEED.
        base = (len(frames) / fps) / (max(0.01, args.speed) * ui_playback_speed)
        tail = max(0, int(args.tail_frames)) / fps
        return max(0.05, base + tail)

    sequence = []
    last_frame_pose = None
    if args.all_good:
        clip_paths = sorted(clips_dir.glob("*.json"))
        if args.shuffle:
            random.shuffle(clip_paths)
        if args.limit > 0:
            clip_paths = clip_paths[: args.limit]
        if not clip_paths:
            raise ValueError(f"No clips found in {clips_dir}")

        current_start = 0.0
        for path in clip_paths:
            duration = read_clip_duration(path)
            clip_name = path.stem
            sequence.append(
                SignEvent(
                    token=clip_name.upper(),
                    clip=clip_name,
                    start=current_start,
                    duration=duration,
                )
            )
            current_start += duration
    else:
        if not args.clip:
            raise ValueError("Provide a clip path/name or use --all-good")

        raw = Path(args.clip)
        if raw.is_absolute():
            clip_path = raw
        elif raw.exists():
            clip_path = raw.resolve()
        else:
            candidate = (clips_dir / raw.name).resolve()
            if not candidate.exists() and raw.suffix != ".json":
                candidate = (clips_dir / f"{raw.name}.json").resolve()
            clip_path = candidate

        if not clip_path.exists():
            raise FileNotFoundError(f"Clip not found: {clip_path}")
        if clip_path.parent != clips_dir:
            raise ValueError(
                "Clip must be inside ui/animation/clips to use the exact UI renderer. "
                f"Expected parent: {clips_dir}, got: {clip_path.parent}"
            )

        duration = read_clip_duration(clip_path)
        clip_name = clip_path.stem
        with clip_path.open("r", encoding="utf-8") as f:
            clip_data = json.load(f)
        frames = clip_data.get("frames", [])
        if isinstance(frames, list) and frames:
            last_frame_pose = frames[-1]
        sequence = [
            SignEvent(token=clip_name.upper(), clip=clip_name, start=0.0, duration=duration)
        ]

    app = QApplication(sys.argv)
    view = view_cls()
    view.disable_live_pose()
    label = "all-good" if args.all_good else sequence[0].clip
    view.setWindowTitle(f"UI Clip Player - {label} ({view_module})")
    view.resize(980, 720)

    def start_playback():
        view.disable_live_pose()
        view.play(sequence)

        if (not args.all_good) and args.return_seconds > 0 and isinstance(last_frame_pose, dict):
            def _build_rest_target_pose(source_pose: dict) -> dict:
                target = dict(source_pose)
                rest = POSES.get("REST", {})
                target.update(rest)

                # Move detailed hand landmarks with the wrist toward REST so
                # hands visibly lower instead of disappearing abruptly.
                left_src = source_pose.get("hand_left")
                right_src = source_pose.get("hand_right")
                left_dst = rest.get("hand_left")
                right_dst = rest.get("hand_right")

                if left_src and left_dst:
                    dx = float(left_dst[0]) - float(left_src[0])
                    dy = float(left_dst[1]) - float(left_src[1])
                    for i in range(21):
                        k = f"left_hand_{i}"
                        if k in source_pose and isinstance(source_pose[k], (list, tuple)) and len(source_pose[k]) >= 2:
                            px, py = source_pose[k][0], source_pose[k][1]
                            target[k] = (float(px) + dx, float(py) + dy)

                if right_src and right_dst:
                    dx = float(right_dst[0]) - float(right_src[0])
                    dy = float(right_dst[1]) - float(right_src[1])
                    for i in range(21):
                        k = f"right_hand_{i}"
                        if k in source_pose and isinstance(source_pose[k], (list, tuple)) and len(source_pose[k]) >= 2:
                            px, py = source_pose[k][0], source_pose[k][1]
                            target[k] = (float(px) + dx, float(py) + dy)
                return target

            def _start_return_motion():
                source_pose = last_frame_pose
                target_pose = _build_rest_target_pose(source_pose)
                steps = max(1, int(args.return_seconds * 60))
                state = {"i": 0}

                view.enable_live_pose()
                return_timer = QTimer(view)

                def _tick():
                    i = state["i"]
                    t = min(1.0, i / float(steps))
                    pose = view_cls.interpolate_pose(source_pose, target_pose, t)
                    view.set_live_pose(pose)
                    state["i"] = i + 1
                    if state["i"] > steps:
                        return_timer.stop()

                return_timer.timeout.connect(_tick)
                return_timer.start(16)

            QTimer.singleShot(max(0, int(sum(evt.duration for evt in sequence) * 1000)), _start_return_motion)

    start_playback()
    view.show()

    if args.loop:
        total_duration = sum(evt.duration for evt in sequence)
        loop_timer = QTimer(view)
        loop_timer.timeout.connect(start_playback)
        loop_timer.start(max(20, int(total_duration * 1000)))

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
