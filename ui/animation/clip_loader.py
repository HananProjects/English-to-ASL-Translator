import json
from pathlib import Path

CLIP_DIR = Path(__file__).parent / "clips"
DEMO_CLIP_DIR = Path(__file__).parent / "clips_demo"
_active_clip_dir = CLIP_DIR


def _normalized_clip_name(sign: str) -> str:
    return str(sign or "").strip().lower()


def resolve_clip_path(sign: str) -> Path:
    clip_name = _normalized_clip_name(sign)
    return _active_clip_dir / f"{clip_name}.json"


def set_demo_mode(enabled: bool) -> Path:
    global _active_clip_dir
    if enabled and DEMO_CLIP_DIR.exists() and any(DEMO_CLIP_DIR.glob("*.json")):
        _active_clip_dir = DEMO_CLIP_DIR
    else:
        _active_clip_dir = CLIP_DIR
    return _active_clip_dir


def is_demo_clip_source_active() -> bool:
    return _active_clip_dir == DEMO_CLIP_DIR


def get_active_clip_dir() -> Path:
    return _active_clip_dir


def load_clip(sign: str):
    path = resolve_clip_path(sign)
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
