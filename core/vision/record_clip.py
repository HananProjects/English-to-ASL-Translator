import argparse
import json
import time
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
    return parser.parse_args()


def _default_clip_path(sign: str) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "ui" / "animation" / "clips" / f"{sign.lower()}.json"


def main():
    args = _parse_args()
    current_sign = args.sign.strip().lower()
    clip_path = _default_clip_path(current_sign)
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

    print("Controls: [R]ecord toggle  [S]ave  [C]lear  [N]ext sign  [Q]uit")
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
            f"{current_sign.upper()} | {status} | frames={len(frames)} | fps={args.fps}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0) if recording else (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            "R:toggle S:save C:clear N:next Q:quit",
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
            if not frames:
                print("No frames recorded yet.")
                continue
            with open(clip_path, "w", encoding="utf-8") as f:
                json.dump({"fps": args.fps, "frames": frames}, f, indent=2)
            print(f"Saved {len(frames)} frames to {clip_path}")
        elif key == ord("n"):
            recording = False
            if frames:
                print("Switching sign: clearing unsaved frames.")
            frames.clear()
            next_sign = input("Enter next sign name: ").strip().lower()
            if not next_sign:
                print(f"Sign unchanged: {current_sign}")
                continue
            current_sign = next_sign
            clip_path = _default_clip_path(current_sign)
            clip_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Target clip switched to: {clip_path}")
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
