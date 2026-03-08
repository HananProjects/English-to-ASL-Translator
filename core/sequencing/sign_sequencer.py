import json
from typing import List, Dict, Optional
from core.english_to_asl.dictionary.asl_signs import ASL_SIGNS
from ui.animation.clip_loader import resolve_clip_path

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
# English->ASL playback uses a slowed visual speed in ASLAnimationView.
# Compensate schedule duration so clips can complete before switching signs.
PLAYBACK_SPEED_FACTOR = 0.62
MAX_PLAYBACK_DURATION = 5.0
_DURATION_CACHE: Dict[str, float] = {}
_VARIANT_INDEX: Dict[str, int] = {}


def _clip_duration_seconds(clip_name: str) -> Optional[float]:
    clip_path = resolve_clip_path(clip_name)
    cache_key = str(clip_path.resolve())
    if cache_key in _DURATION_CACHE:
        return _DURATION_CACHE[cache_key]
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
        _DURATION_CACHE[cache_key] = duration
        return duration
    except Exception:
        return None


def _select_clip_name(token: str, sign_def: Dict) -> Optional[str]:
    clips = sign_def.get("clips")
    if isinstance(clips, list) and clips:
        idx = _VARIANT_INDEX.get(token, 0)
        clip_name = clips[idx % len(clips)]
        _VARIANT_INDEX[token] = idx + 1
        return clip_name
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
            # Fingerspelling fallback: allow single-letter tokens to play
            # directly from lowercase clip names (e.g., A -> a.json).
            token_text = str(token or "").strip()
            if len(token_text) == 1 and token_text.isalpha():
                fallback_clip = token_text.lower()
                clip_path = resolve_clip_path(fallback_clip)
                if clip_path.exists():
                    sign_def = {"clip": fallback_clip, "duration": DEFAULT_SIGN_DURATION}
                else:
                    print(f"[WARN] No ASL sign metadata for token: {token}")
                    continue
            else:
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
            # Ensure clip can fully play at the slowed playback speed.
            compensated_duration = clip_duration / max(PLAYBACK_SPEED_FACTOR, 1e-6)
            duration = max(configured_duration, compensated_duration)
            # Keep very long clips bounded so sentences remain responsive.
            duration = min(duration, MAX_PLAYBACK_DURATION)

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
