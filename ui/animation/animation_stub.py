import time
from typing import List, Optional
from core.sequencing.sign_sequencer import SignEvent


class AnimationStub:
    """
    Time-driven stub for ASL sign playback.
    No visuals — only sequencing verification.
    """

    def __init__(self):
        self.sequence: List[SignEvent] = []
        self.start_time: Optional[float] = None
        self.current_index = 0

    def play(self, sequence: List[SignEvent]):
        self.sequence = sequence
        self.start_time = time.time()
        self.current_index = 0

        if sequence:
            print("[ANIM] Sequence start:", [e.sign for e in sequence])

    def update(self):
        if not self.sequence or self.start_time is None:
            return

        elapsed = time.time() - self.start_time

        while self.current_index < len(self.sequence):
            event = self.sequence[self.current_index]

            if elapsed >= event.start:
                print(f"[ANIM] Playing sign: {event.sign}")
                self.current_index += 1
            else:
                break
