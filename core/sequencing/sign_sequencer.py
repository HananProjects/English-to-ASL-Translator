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

        duration = sign_def.get("duration", DEFAULT_SIGN_DURATION)

        events.append(
            SignEvent(
                token=token,
                clip=sign_def["clip"],
                start=current_time,
                duration=duration
            )
        )

        current_time += duration


    return events