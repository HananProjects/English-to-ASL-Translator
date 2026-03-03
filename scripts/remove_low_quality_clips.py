import json
import os
import shutil

THRESHOLD = 0.55

REPORT_PATH = "data/eval/sign_quality_report.json"
CLIPS_DIR = "ui/animation/clips"
REJECTED_DIR = "ui/animation/clips_rejected"

os.makedirs(REJECTED_DIR, exist_ok=True)

with open(REPORT_PATH, "r") as f:
    report = json.load(f)

bottom_clips = report.get("bottom_clips", [])

moved = 0

for clip_info in bottom_clips:
    score = clip_info["score"]
    clip_name = clip_info["clip"]

    if score < THRESHOLD:
        src = os.path.join(CLIPS_DIR, clip_name)
        dst = os.path.join(REJECTED_DIR, clip_name)

        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved {clip_name} (score={score})")
            moved += 1

print(f"\nTotal moved: {moved}")