import sys
import os
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

# KEEP A GLOBAL REFERENCE (IMPORTANT)
window = None

def main():
    global window

    app = QApplication(sys.argv)

    window = MainWindow()
    fullscreen_env = os.getenv("ASL_TOUCH_FULLSCREEN", "auto").strip().lower()
    if fullscreen_env in {"1", "true", "yes", "on"}:
        window.showFullScreen()
    elif fullscreen_env in {"0", "false", "no", "off"}:
        window.show()
    else:
        screen = app.primaryScreen()
        geo = screen.availableGeometry() if screen else None
        if geo is not None and (geo.width() <= 1024 or geo.height() <= 600):
            window.showFullScreen()
        else:
            window.show()

    # BLOCK here until app quits
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
