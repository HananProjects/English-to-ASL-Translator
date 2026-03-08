#!/usr/bin/env python3
"""Run presentation-focused verification checks and output one summary report.

This runner targets the metrics shown on your final presentation slide:
- Text -> Gesture translation accuracy
- Gesture -> Text/label accuracy
- Sign playability coverage
- Text to Speech smoke test
- Speech to Text smoke test
- Optional battery stress soak
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_EVAL_DIR = REPO_ROOT / "data" / "eval"


def _run(cmd: List[str], timeout: int = 300) -> Dict[str, Any]:
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "duration_s": round(time.time() - start, 3),
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": exc.stdout or "",
            "stderr": f"TIMEOUT after {timeout}s",
            "duration_s": round(time.time() - start, 3),
            "cmd": cmd,
        }


def _extract_float(name: str, text: str) -> float | None:
    m = re.search(rf"{re.escape(name)}=([0-9]*\.?[0-9]+)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def run_text_to_gesture_eval(python_bin: str) -> Dict[str, Any]:
    cmd = [python_bin, "scripts/eval_english_to_asl.py", "--eval-file", "data/eval/english_to_asl_eval.json"]
    result = _run(cmd)
    metrics = {
        "exact_match": _extract_float("exact_match", result["stdout"]),
        "token_precision": _extract_float("token_precision", result["stdout"]),
        "token_recall": _extract_float("token_recall", result["stdout"]),
        "token_f1": _extract_float("token_f1", result["stdout"]),
    }
    result["metrics"] = metrics
    return result


def run_gesture_to_text_eval(python_bin: str) -> Dict[str, Any]:
    report_path = DATA_EVAL_DIR / "asl_to_english_eval.json"
    cmd = [python_bin, "scripts/eval_asl_sequence_model.py", "--report", str(report_path)]
    result = _run(cmd)
    metrics: Dict[str, Any] = {"accuracy": None, "total_samples": None, "correct": None}
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            metrics["accuracy"] = data.get("accuracy")
            metrics["total_samples"] = data.get("total_samples")
            metrics["correct"] = data.get("correct")
        except Exception:
            pass
    result["metrics"] = metrics
    result["report_path"] = str(report_path)
    return result


def run_playability_check(python_bin: str) -> Dict[str, Any]:
    report_path = DATA_EVAL_DIR / "sign_playability_report.json"
    cmd = [python_bin, "scripts/check_sign_playability.py", "--report", str(report_path)]
    result = _run(cmd)
    metrics: Dict[str, Any] = {
        "labels_checked": None,
        "playable_count": None,
        "unplayable_count": None,
        "playable_ratio": None,
    }
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            total = int(data.get("labels_checked", 0))
            playable = int(data.get("playable_count", 0))
            unplayable = int(data.get("unplayable_count", 0))
            ratio = (playable / total) if total > 0 else None
            metrics.update(
                {
                    "labels_checked": total,
                    "playable_count": playable,
                    "unplayable_count": unplayable,
                    "playable_ratio": ratio,
                }
            )
        except Exception:
            pass
    result["metrics"] = metrics
    result["report_path"] = str(report_path)
    return result


def run_stt_smoke(python_bin: str) -> Dict[str, Any]:
    cmd = [python_bin, "-m", "pytest", "-q", "tests/test_stt_interface.py", "tests/test_stt_vosk_backend.py"]
    return _run(cmd, timeout=240)


def run_tts_smoke() -> Dict[str, Any]:
    # Keep this smoke test backend-agnostic and lightweight.
    for candidate in ("espeak-ng", "espeak"):
        path = shutil.which(candidate)
        if not path:
            continue
        # Generate and discard audio (no speaker requirement for CI/test rigs).
        cmd = [path, "-q", "Presentation TTS test."]
        return _run(cmd, timeout=30)
    return {
        "ok": False,
        "returncode": 127,
        "stdout": "",
        "stderr": "No espeak/espeak-ng executable found on PATH.",
        "duration_s": 0.0,
        "cmd": [],
    }


def run_battery_stress(python_bin: str, duration_seconds: int) -> Dict[str, Any]:
    cmd = [
        python_bin,
        "scripts/run_battery_stress.py",
        "--duration-seconds",
        str(duration_seconds),
        "--status-interval",
        str(max(10, min(60, duration_seconds // 4 if duration_seconds > 0 else 30))),
    ]
    # Battery stress can run longer by design.
    timeout = max(120, duration_seconds + 60) if duration_seconds > 0 else 3600
    return _run(cmd, timeout=timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run presentation verification checks and emit one JSON summary report."
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter to run checks with (default: current interpreter).",
    )
    parser.add_argument(
        "--output",
        default=str(DATA_EVAL_DIR / "presentation_verification_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--run-battery-stress",
        action="store_true",
        help="Include battery stress test in this run.",
    )
    parser.add_argument(
        "--battery-duration-seconds",
        type=int,
        default=0,
        help="Battery stress duration when --run-battery-stress is set (e.g., 14400 for 4h).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    report: Dict[str, Any] = {
        "started_at": started,
        "python_bin": args.python_bin,
        "offline_expected": True,
        "checks": {},
    }

    print("[presentation] Running text->gesture evaluation...")
    report["checks"]["text_to_gesture_eval"] = run_text_to_gesture_eval(args.python_bin)

    print("[presentation] Running gesture->text evaluation...")
    report["checks"]["gesture_to_text_eval"] = run_gesture_to_text_eval(args.python_bin)

    print("[presentation] Running sign playability check...")
    report["checks"]["sign_playability"] = run_playability_check(args.python_bin)

    print("[presentation] Running speech-to-text smoke test...")
    report["checks"]["speech_to_text_smoke"] = run_stt_smoke(args.python_bin)

    print("[presentation] Running text-to-speech smoke test...")
    report["checks"]["text_to_speech_smoke"] = run_tts_smoke()

    if args.run_battery_stress:
        duration = int(args.battery_duration_seconds)
        print(f"[presentation] Running battery stress for {duration}s...")
        report["checks"]["battery_stress"] = run_battery_stress(args.python_bin, duration)
    else:
        report["checks"]["battery_stress"] = {
            "ok": None,
            "skipped": True,
            "reason": "Use --run-battery-stress to enable this check.",
        }

    report["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["summary"] = {
        "text_to_gesture_exact_match": report["checks"]["text_to_gesture_eval"]["metrics"].get(
            "exact_match"
        ),
        "gesture_to_text_accuracy": report["checks"]["gesture_to_text_eval"]["metrics"].get(
            "accuracy"
        ),
        "playable_sign_ratio": report["checks"]["sign_playability"]["metrics"].get(
            "playable_ratio"
        ),
        "stt_smoke_ok": report["checks"]["speech_to_text_smoke"].get("ok"),
        "tts_smoke_ok": report["checks"]["text_to_speech_smoke"].get("ok"),
        "battery_stress_ok": report["checks"]["battery_stress"].get("ok"),
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[presentation] Report written to: {out_path}")
    print("[presentation] Summary:")
    for key, value in report["summary"].items():
        print(f"  - {key}: {value}")

    # Return non-zero only if a non-skipped check failed.
    failed = []
    for name, result in report["checks"].items():
        if result.get("skipped"):
            continue
        if result.get("ok") is False:
            failed.append(name)
    if failed:
        print(f"[presentation] Failed checks: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

