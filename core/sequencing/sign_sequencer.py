from typing import List, Dict


class SignEvent:
    """
    Represents a single ASL sign scheduled in time.
    """
    def __init__(self, sign: str, start: float, duration: float):
        self.sign = sign
        self.start = start
        self.duration = duration

    def to_dict(self) -> Dict:
        return {
            "sign": self.sign,
            "start": self.start,
            "duration": self.duration
        }


# Base duration per sign (seconds)
DEFAULT_SIGN_DURATION = 0.7

# Optional per-sign overrides
SIGN_DURATIONS = {
    "YOU": 0.6,
    "GO": 0.8,
    "WHERE": 0.7,
}


def sequence_signs(tokens: List[str]) -> List[SignEvent]:
    """
    Convert ASL tokens into a time-ordered sign sequence.
    """
    events: List[SignEvent] = []
    current_time = 0.0

    for token in tokens:
        duration = SIGN_DURATIONS.get(token, DEFAULT_SIGN_DURATION)

        event = SignEvent(
            sign=token,
            start=current_time,
            duration=duration
        )

        events.append(event)
        current_time += duration

    return events