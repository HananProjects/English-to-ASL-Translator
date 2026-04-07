import json
from pathlib import Path

CLIP_DIR = Path(__file__).parent / "clips"


def _normalized_clip_name(sign: str) -> str:
    return str(sign or "").strip().lower()


def resolve_clip_path(sign: str) -> Path:
    clip_name = _normalized_clip_name(sign)
    return CLIP_DIR / f"{clip_name}.json"


def load_clip(sign: str):
    path = resolve_clip_path(sign)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
