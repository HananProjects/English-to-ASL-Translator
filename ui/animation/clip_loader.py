import json
from pathlib import Path

CLIP_DIR = Path(__file__).parent / "clips"


def load_clip(sign: str):
    path = CLIP_DIR / f"{sign.lower()}.json"
    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)