import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

# KEEP A GLOBAL REFERENCE (IMPORTANT)
window = None

def main():
    global window

    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    # BLOCK here until app quits
    sys.exit(app.exec())

if __name__ == "__main__":
    main()