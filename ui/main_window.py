from core.engine import english_to_asl
from core.mic_utils import record_audio
from PySide6.QtCore import QThread, Qt, QTimer
from core.sequencing.sign_sequencer import sequence_signs
from ui.animation.animation_stub import AnimationStub
from ui.worker import TranslationWorker
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton
)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        # Fullscreen kiosk mode
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.showFullScreen()

        self.setWindowTitle("English → ASL Translator")
        self.setMinimumSize(800, 480)

        layout = QVBoxLayout()

        self.status_label = QLabel("Status: Idle")
        self.status_label.setStyleSheet("font-size: 18px;")

        self.tokens_label = QLabel("ASL Output:")
        self.tokens_label.setStyleSheet("font-size: 22px;")

        self.record_button = QPushButton("Record")
        self.record_button.setStyleSheet("font-size: 20px; height: 60px;")
        self.record_button.clicked.connect(self.on_record_clicked)

        layout.addWidget(self.status_label)
        layout.addWidget(self.tokens_label)
        layout.addWidget(self.record_button)

        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(30)
        self.setLayout(layout)

        # Animation stub
        self.anim = AnimationStub()

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.anim.update)
        self.anim_timer.start(50)

    def on_record_clicked(self):
        self.status_label.setText("Status: Recording...")
        self.tokens_label.setText("ASL Output:")

        self.thread = QThread()
        self.worker = TranslationWorker()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_translation_finished)
        self.worker.error.connect(self.on_translation_error)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.start()

    def on_translation_finished(self, data):
        tokens = data["tokens"]

        self.tokens_label.setText(f"ASL Output: {' '.join(tokens)}")
        self.status_label.setText(
            f"Confidence: {data['confidence']:.2f} | "
            f"Latency: {data['latency']} ms"
        )

        # Sequencing → animation stub
        sequence = sequence_signs(tokens)
        self.anim.play(sequence)

    def on_translation_error(self, message):
        self.status_label.setText(f"Error: {message}")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()