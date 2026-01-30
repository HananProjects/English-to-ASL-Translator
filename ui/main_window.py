from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QPushButton
)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("English → ASL Translator")
        self.setMinimumSize(800, 480)  # Pi touchscreen friendly

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

        self.setLayout(layout)

    def on_record_clicked(self):
        # Placeholder for now
        self.status_label.setText("Status: Recording (stub)")
        self.tokens_label.setText("ASL Output: YOU GO WHERE")