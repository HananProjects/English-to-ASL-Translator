import json
from pathlib import Path
from typing import List, Dict
from core.english_to_asl.dictionary.asl_signs import ASL_SIGNS

class SignEvent:
    """
    Represents a single ASL sign scheduled in time.
    """
    def __init__(self, token: str, clip: str, start: float, duration: float):
        self.token = token
        self.clip = clip
        self.start = start
        self.duration = duration

    def to_dict(self) -> Dict:
        return {
            "token": self.token,
            "clip": self.clip,
            "start": self.start,
            "duration": self.duration
        }

# Base duration per sign (seconds)
DEFAULT_SIGN_DURATION = 0.7
MAX_PLAYBACK_DURATION = 2.5
_DURATION_CACHE: Dict[str, float] = {}


def _clip_duration_seconds(clip_name: str) -> float | None:
    if clip_name in _DURATION_CACHE:
        return _DURATION_CACHE[clip_name]

    repo_root = Path(__file__).resolve().parents[2]
    clip_path = repo_root / "ui" / "animation" / "clips" / f"{clip_name}.json"
    if not clip_path.exists():
        return None

    try:
        with open(clip_path, "r", encoding="utf-8") as f:
            clip_data = json.load(f)
        fps = float(clip_data.get("fps", 0))
        frames = clip_data.get("frames", [])
        frame_count = len(frames) if isinstance(frames, list) else 0
        if fps <= 0 or frame_count <= 0:
            return None
        duration = frame_count / fps
        _DURATION_CACHE[clip_name] = duration
        return duration
    except Exception:
        return None


def _infer_avg_clip_name(clips: List[str]) -> str | None:
    if not clips:
        return None
    first = clips[0]
    if "_" in first:
        base = first.rsplit("_", 1)[0]
    else:
        base = first
    return f"{base}_avg"


def _select_clip_name(token: str, sign_def: Dict) -> str | None:
    clips = sign_def.get("clips")
    if isinstance(clips, list) and clips:
        merged_clip = sign_def.get("merged_clip")
        if isinstance(merged_clip, str) and merged_clip:
            return merged_clip
        inferred_avg = _infer_avg_clip_name(clips)
        if inferred_avg and _clip_duration_seconds(inferred_avg) is not None:
            return inferred_avg
        # Stable fallback: do not round-robin variants by default.
        return clips[0]
    return sign_def.get("clip")

def sequence_signs(tokens: List[str]) -> List[SignEvent]:
    """
    Convert ASL tokens into a time-ordered sign sequence.
    """
    events: List[SignEvent] = []
    current_time = 0.0

    for token in tokens:
        sign_def = ASL_SIGNS.get(token)

        if not sign_def:
            print(f"[WARN] No ASL sign metadata for token: {token}")
            continue

        clip_name = _select_clip_name(token, sign_def)
        if not clip_name:
            print(f"[WARN] No clip name configured for token: {token}")
            continue
        clip_duration = _clip_duration_seconds(clip_name)
        configured_duration = float(sign_def.get("duration", DEFAULT_SIGN_DURATION))
        if clip_duration is None:
            duration = configured_duration
        else:
            # Prefer actual clip timing for natural motion, but cap very long
            # clips to keep sentence playback responsive.
            duration = min(clip_duration, max(configured_duration, MAX_PLAYBACK_DURATION))

        events.append(
            SignEvent(
                token=token,
                clip=clip_name,
                start=current_time,
                duration=duration
            )
        )

        current_time += duration


    return events
