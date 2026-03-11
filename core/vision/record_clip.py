import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp

from core.vision.pose_adapter import mediapipe_to_pose_dict


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Record MediaPipe pose/hand landmarks as an ASL clip JSON."
    )
    parser.add_argument("sign", help="Sign token/clip name, e.g. hello")
    parser.add_argument("--fps", type=int, default=12, help="Output clip FPS")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index"
    )
    parser.add_argument(
        "--allow-no-hand",
        action="store_true",
        help="Record frames even when no hand landmarks are detected."
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Optional session tag stored in clip metadata, e.g. prof_demo_round1",
    )
    parser.add_argument(
        "--signer-id",
        default="",
        help="Optional signer tag stored in clip metadata, e.g. hanan",
    )
    return parser.parse_args()


def _clips_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "ui" / "animation" / "clips"


def _split_sign_name(sign: str) -> tuple[str, int | None]:
    m = re.match(r"^(.*?)(?:_(\d+))?$", sign.strip().lower())
    if not m:
        return sign.strip().lower(), None
    base = (m.group(1) or sign).strip("_")
    num = m.group(2)
    return base, (int(num) if num is not None else None)


def _resolve_clip_name(sign: str) -> str:
    clips_dir = _clips_dir()
    clips_dir.mkdir(parents=True, exist_ok=True)
    base, explicit_num = _split_sign_name(sign)
    if explicit_num is not None:
        return f"{base}_{explicit_num}"

    pattern = re.compile(rf"^{re.escape(base)}_(\d+)\.json$")
    highest = 0
    for path in clips_dir.glob(f"{base}_*.json"):
        match = pattern.match(path.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"{base}_{highest + 1}"


def _clip_path_for_name(clip_name: str) -> Path:
    return _clips_dir() / f"{clip_name.lower()}.json"


def _next_clip_name_for_same_sign(current_clip_name: str) -> str:
    base, num = _split_sign_name(current_clip_name)
    return f"{base}_{(num or 1) + 1}"


def _save_clip(
    clip_path: Path,
    fps: int,
    frames: list,
    sign_label: str,
    clip_name: str,
    session_id: str,
    signer_id: str,
) -> bool:
    if not frames:
        print("No frames recorded yet.")
        return False
    metadata = {
        "label": sign_label.upper(),
        "clip_name": clip_name,
        "session_id": session_id or clip_name,
        "signer_id": signer_id or "unknown",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "record_clip",
    }
    with open(clip_path, "w", encoding="utf-8") as f:
        json.dump({"fps": fps, "frames": frames, "metadata": metadata}, f, indent=2)
    print(f"Saved {len(frames)} frames to {clip_path}")
    return True


def main():
    args = _parse_args()
    current_label = args.sign.strip().lower()
    current_clip_name = _resolve_clip_name(current_label)
    clip_path = _clip_path_for_name(current_clip_name)
    clip_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    draw = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic

    recording = False
    frames = []
    last_capture_time = 0.0
    sample_period = 1.0 / max(1, args.fps)

    print("Controls: [R]ecord toggle  [S]ave  [I] save+increment  [C]lear  [N]ext sign  [Q]uit")
    print(f"Target clip: {clip_path}")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = holistic.process(rgb)

        pose_landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
        left_hand_landmarks = result.left_hand_landmarks.landmark if result.left_hand_landmarks else None
        right_hand_landmarks = result.right_hand_landmarks.landmark if result.right_hand_landmarks else None

        hand_detected = bool(left_hand_landmarks or right_hand_landmarks)

        if recording and pose_landmarks and (hand_detected or args.allow_no_hand):
            now = time.time()
            if now - last_capture_time >= sample_period:
                pose = mediapipe_to_pose_dict(
                    pose_landmarks,
                    left_hand_landmarks,
                    right_hand_landmarks,
                )
                frames.append(pose)
                last_capture_time = now

        if result.pose_landmarks:
            draw.draw_landmarks(
                frame,
                result.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
            )
        if result.left_hand_landmarks:
            draw.draw_landmarks(
                frame,
                result.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
            )
        if result.right_hand_landmarks:
            draw.draw_landmarks(
                frame,
                result.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
            )

        status = "RECORDING" if recording else "IDLE"
        cv2.putText(
            frame,
            f"{current_clip_name.upper()} | {status} | frames={len(frames)} | fps={args.fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if recording else (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "R:toggle S:save I:save+inc C:clear N:next Q:quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Hands L={bool(left_hand_landmarks)} R={bool(right_hand_landmarks)}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0) if hand_detected else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("ASL Clip Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            recording = not recording
            if recording:
                last_capture_time = 0.0
        elif key == ord("c"):
            frames.clear()
            print("Cleared frames.")
        elif key == ord("s"):
            _save_clip(
                clip_path,
                args.fps,
                frames,
                sign_label=current_label,
                clip_name=current_clip_name,
                session_id=args.session_id.strip(),
                signer_id=args.signer_id.strip(),
            )
        elif key == ord("i"):
            if not _save_clip(
                clip_path,
                args.fps,
                frames,
                sign_label=current_label,
                clip_name=current_clip_name,
                session_id=args.session_id.strip(),
                signer_id=args.signer_id.strip(),
            ):
                continue
            frames.clear()
            recording = False
            current_clip_name = _next_clip_name_for_same_sign(current_clip_name)
            clip_path = _clip_path_for_name(current_clip_name)
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Auto-incremented target clip: {clip_path}")
        elif key == ord("n"):
            recording = False
            if frames:
                print("Switching sign: clearing unsaved frames.")
            frames.clear()
            next_sign = input("Enter next sign name: ").strip().lower()
            if not next_sign:
                print(f"Sign unchanged: {current_label}")
                continue
            current_label = next_sign
            current_clip_name = _resolve_clip_name(current_label)
            clip_path = _clip_path_for_name(current_clip_name)
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Target clip switched to: {clip_path}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
