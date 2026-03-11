#!/usr/bin/env python3
import argparse
import os
import signal
import sys
import time

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

from core.asl_to_english.recognizer import SignStreamRecognizer
from core.engine import asl_to_english, english_to_asl
from core.vision.live_pose import LivePoseFilter
from core.vision.pose_adapter import mediapipe_to_pose_dict


TEXT_SAMPLES = [
    "hello how are you",
    "where are you going",
    "please help me",
    "thank you very much",
    "what is your name",
    "i need water",
    "can you repeat that",
    "i am ready now",
]


def env_int(name: str, default: int, min_value: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    if min_value is not None and value < min_value:
        return min_value
    return value


def env_float(name: str, default: float, min_value: float | None = None) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except Exception:
        return default
    if min_value is not None and value < min_value:
        return min_value
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Headless battery soak test for continuous ASL camera processing. "
            "This mirrors the UI camera pipeline without the Qt interface."
        )
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=0,
        help="Stop automatically after this many seconds. 0 means run until interrupted.",
    )
    parser.add_argument(
        "--status-interval",
        type=int,
        default=30,
        help="Print summary stats every N seconds.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=env_int("ASL_CAMERA_INDEX", 0, min_value=0),
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=env_int("ASL_CAMERA_WIDTH", 1280, min_value=160),
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=env_int("ASL_CAMERA_HEIGHT", 720, min_value=120),
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=env_int("ASL_CAMERA_FPS", 30, min_value=1),
    )
    parser.add_argument(
        "--camera-buffer-size",
        type=int,
        default=env_int("ASL_CAMERA_BUFFER_SIZE", 1, min_value=1),
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=env_int("ASL_PROCESS_WIDTH", 640, min_value=160),
    )
    parser.add_argument(
        "--process-height",
        type=int,
        default=env_int("ASL_PROCESS_HEIGHT", 360, min_value=120),
    )
    parser.add_argument(
        "--process-every-n",
        type=int,
        default=env_int("ASL_PROCESS_EVERY_N", 1, min_value=1),
    )
    parser.add_argument(
        "--camera-zoom",
        type=float,
        default=env_float("ASL_CAMERA_ZOOM", 1.0, min_value=1.0),
    )
    parser.add_argument(
        "--max-failed-reads",
        type=int,
        default=600,
        help="Exit after this many consecutive failed camera reads.",
    )
    parser.add_argument(
        "--text-roundtrip-every",
        type=int,
        default=0,
        help=(
            "Run an extra English->ASL->English text roundtrip every N processed frames. "
            "0 disables the extra CPU load."
        ),
    )
    return parser


def format_rate(count: int, elapsed: float) -> float:
    if elapsed <= 0:
        return 0.0
    return count / elapsed


def main() -> int:
    if cv2 is None or mp is None:
        print("Error: missing camera dependencies (cv2 and/or mediapipe).", file=sys.stderr)
        return 1

    parser = build_parser()
    args = parser.parse_args()

    running = True

    def request_stop(signum, _frame):
        nonlocal running
        running = False
        print(f"\n[stress] received signal {signum}, stopping...")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    print(
        "[stress] camera="
        f"{args.camera_index} {args.camera_width}x{args.camera_height}@{args.camera_fps} "
        f"buffer={args.camera_buffer_size} proc={args.process_width}x{args.process_height} "
        f"skip={args.process_every_n} zoom={args.camera_zoom:.2f} "
        f"text_roundtrip_every={args.text_roundtrip_every or 'off'} "
        f"duration={args.duration_seconds or 'until-stopped'}s"
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        print(
            f"Error: unable to open camera index {args.camera_index}. "
            "Try a different --camera-index or set ASL_CAMERA_INDEX.",
            file=sys.stderr,
        )
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, args.camera_buffer_size)
    except Exception:
        pass

    recognizer = SignStreamRecognizer(
        prefer_model=True,
        stable_frames=6,
        min_confidence=0.60,
        emit_cooldown_frames=10,
        pause_frames=18,
    )
    print(f"[stress] recognizer matcher={type(recognizer.matcher).__name__}")
    pose_filter = LivePoseFilter(
        alpha=env_float("ASL_POSE_SMOOTHING", 0.50, min_value=0.0),
        hand_hold_frames=env_int("ASL_HAND_HOLD_FRAMES", 3, min_value=0),
        body_hold_frames=env_int("ASL_BODY_HOLD_FRAMES", 1, min_value=0),
    )

    holistic = mp.solutions.holistic.Holistic(
        model_complexity=0,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    start_time = time.time()
    next_status_time = start_time + max(1, args.status_interval)
    frame_count = 0
    processed_frames = 0
    failed_reads = 0
    detected_tokens = 0
    completed_sentences = 0
    text_roundtrips = 0
    last_sentence = ""
    last_roundtrip = ""

    try:
        while running:
            now = time.time()
            if args.duration_seconds > 0 and (now - start_time) >= args.duration_seconds:
                print("[stress] duration reached, stopping...")
                break

            ok, frame = cap.read()
            if not ok:
                failed_reads += 1
                if failed_reads >= args.max_failed_reads:
                    print(
                        f"[stress] camera read failed {failed_reads} times in a row, exiting.",
                        file=sys.stderr,
                    )
                    return 2
                time.sleep(0.005)
                continue

            failed_reads = 0
            frame_count += 1

            if args.camera_zoom > 1.0:
                height, width = frame.shape[:2]
                crop_width = max(2, int(width / args.camera_zoom))
                crop_height = max(2, int(height / args.camera_zoom))
                x0 = (width - crop_width) // 2
                y0 = (height - crop_height) // 2
                frame = frame[y0:y0 + crop_height, x0:x0 + crop_width]

            if args.process_every_n > 1 and (frame_count % args.process_every_n) != 0:
                if now >= next_status_time:
                    elapsed = now - start_time
                    print(
                        "[stress] uptime="
                        f"{elapsed:.1f}s frames={frame_count} processed={processed_frames} "
                        f"capture_fps={format_rate(frame_count, elapsed):.2f} "
                        f"process_fps={format_rate(processed_frames, elapsed):.2f} "
                        f"tokens={detected_tokens} sentences={completed_sentences} "
                        f"text_roundtrips={text_roundtrips}"
                    )
                    next_status_time = now + max(1, args.status_interval)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proc = cv2.resize(
                rgb,
                (args.process_width, args.process_height),
                interpolation=cv2.INTER_AREA,
            )
            result = holistic.process(proc)
            processed_frames += 1

            pose_landmarks = result.pose_landmarks.landmark if result.pose_landmarks else None
            left_hand_landmarks = (
                result.left_hand_landmarks.landmark if result.left_hand_landmarks else None
            )
            right_hand_landmarks = (
                result.right_hand_landmarks.landmark if result.right_hand_landmarks else None
            )

            if pose_landmarks is not None:
                pose = mediapipe_to_pose_dict(
                    pose_landmarks,
                    left_hand_landmarks,
                    right_hand_landmarks,
                )
                pose = pose_filter.apply(pose)
                update = recognizer.process(pose)
                if update.detected_token is not None:
                    detected_tokens += 1
                    print(
                        f"[stress] token={update.detected_token} "
                        f"conf={update.confidence:.2f}"
                    )
                if update.sentence_tokens is not None:
                    completed_sentences += 1
                    reverse = asl_to_english(tokens=update.sentence_tokens)
                    last_sentence = " ".join(update.sentence_tokens)
                    print(
                        "[stress] sentence="
                        f"{last_sentence} -> {reverse.english_text!r} "
                        f"(conf={reverse.confidence:.2f})"
                    )

            if (
                args.text_roundtrip_every > 0
                and processed_frames % args.text_roundtrip_every == 0
            ):
                sample = TEXT_SAMPLES[text_roundtrips % len(TEXT_SAMPLES)]
                forward = english_to_asl(text=sample)
                reverse = asl_to_english(tokens=forward.asl_tokens)
                text_roundtrips += 1
                last_roundtrip = reverse.english_text
                print(
                    "[stress] roundtrip="
                    f"{sample!r} -> {forward.asl_tokens} -> {reverse.english_text!r}"
                )

            if now >= next_status_time:
                elapsed = now - start_time
                print(
                    "[stress] uptime="
                    f"{elapsed:.1f}s frames={frame_count} processed={processed_frames} "
                    f"capture_fps={format_rate(frame_count, elapsed):.2f} "
                    f"process_fps={format_rate(processed_frames, elapsed):.2f} "
                    f"tokens={detected_tokens} sentences={completed_sentences} "
                    f"text_roundtrips={text_roundtrips}"
                )
                next_status_time = now + max(1, args.status_interval)
    finally:
        try:
            holistic.close()
        except Exception:
            pass
        cap.release()

    elapsed = max(0.0, time.time() - start_time)
    print("[stress] complete")
    print(
        "[stress] summary "
        f"uptime={elapsed:.1f}s "
        f"frames={frame_count} "
        f"processed={processed_frames} "
        f"capture_fps={format_rate(frame_count, elapsed):.2f} "
        f"process_fps={format_rate(processed_frames, elapsed):.2f} "
        f"tokens={detected_tokens} "
        f"sentences={completed_sentences} "
        f"text_roundtrips={text_roundtrips}"
    )
    if last_sentence:
        print(f"[stress] last_sentence={last_sentence}")
    if last_roundtrip:
        print(f"[stress] last_roundtrip={last_roundtrip!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
