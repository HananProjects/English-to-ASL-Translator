import os
import shutil
import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

# KEEP A GLOBAL REFERENCE (IMPORTANT)
window = None


def _maybe_reexec_with_libcamerify() -> None:
    if os.getenv("ASL_LIBCAMERIFY_ACTIVE", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    if os.getenv("ASL_DISABLE_LIBCAMERIFY", "").strip().lower() in {"1", "true", "yes", "on"}:
        return
    if sys.platform != "linux":
        return

    libcamerify = shutil.which("libcamerify")
    if not libcamerify:
        return

    os.environ["ASL_LIBCAMERIFY_ACTIVE"] = "1"
    os.execvp(libcamerify, [libcamerify, sys.executable, "-m", "ui.main"])

def main():
    global window
    _maybe_reexec_with_libcamerify()

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
