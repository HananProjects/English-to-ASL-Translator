import sys

from PySide6.QtWidgets import QApplication

from ui.clip_reviewer_window import ClipReviewerWindow


window = None


def main():
    global window
    app = QApplication(sys.argv)
    window = ClipReviewerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
