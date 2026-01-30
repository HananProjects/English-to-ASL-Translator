import time
from typing import List
from PySide6.QtWidgets import QWidget
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QFont
from core.sequencing.sign_sequencer import SignEvent


class ASLAnimationView(QWidget):
    """
    Visual animation view driven by ASL sign sequencing.
    (Stub visuals — real animation comes later)
    """

    def __init__(self):
        super().__init__()

        self.sequence: List[SignEvent] = []
        self.start_time: float | None = None
        self.current_sign: str | None = None

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(30)

    def play(self, sequence: List[SignEvent]):
        self.sequence = sequence
        self.start_time = time.time()
        self.current_sign = None

    def paintEvent(self, event):
        if not self.sequence or self.start_time is None:
            return

        elapsed = time.time() - self.start_time

        active = None
        for e in self.sequence:
            if e.start <= elapsed < e.start + e.duration:
                active = e.sign
                break

        if active is None:
            return

        self.current_sign = active

        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setFont(QFont("Arial", 48))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                f"SIGNING: {active}"
            )
        finally:
            painter.end()