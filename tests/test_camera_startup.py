from types import SimpleNamespace

import ui.main
import ui.worker_camera


def test_maybe_reexec_with_libcamerify_skips_when_disabled(monkeypatch):
    monkeypatch.setenv("ASL_DISABLE_LIBCAMERIFY", "1")
    monkeypatch.delenv("ASL_LIBCAMERIFY_ACTIVE", raising=False)
    monkeypatch.setattr(ui.main.sys, "platform", "linux")
    monkeypatch.setattr(ui.main.shutil, "which", lambda name: "/usr/bin/libcamerify")

    called = {"execvp": False}

    def fake_execvp(*args):
        called["execvp"] = True

    monkeypatch.setattr(ui.main.os, "execvp", fake_execvp)

    ui.main._maybe_reexec_with_libcamerify()

    assert called["execvp"] is False


def test_camera_probe_tries_requested_index_then_fallbacks(monkeypatch):
    attempts = []

    class FakeCapture:
        def __init__(self, index):
            self.index = index

        def isOpened(self):
            return self.index == 2

        def release(self):
            attempts.append(f"release:{self.index}")

    monkeypatch.setattr(
        ui.worker_camera,
        "cv2",
        SimpleNamespace(VideoCapture=lambda index: attempts.append(index) or FakeCapture(index)),
    )

    cap, detected_index, attempted = ui.worker_camera._open_camera_capture(3, 4)

    assert cap is not None
    assert detected_index == 2
    assert attempted == [3, 0, 1, 2]
    assert attempts == [3, "release:3", 0, "release:0", 1, "release:1", 2]


def test_camera_probe_reports_failure_when_no_index_opens(monkeypatch):
    attempts = []

    class FakeCapture:
        def __init__(self, index):
            self.index = index

        def isOpened(self):
            return False

        def release(self):
            attempts.append(f"release:{self.index}")

    monkeypatch.setattr(
        ui.worker_camera,
        "cv2",
        SimpleNamespace(VideoCapture=lambda index: attempts.append(index) or FakeCapture(index)),
    )

    cap, detected_index, attempted = ui.worker_camera._open_camera_capture(0, 3)

    assert cap is None
    assert detected_index is None
    assert attempted == [0, 1, 2]
    assert attempts == [0, "release:0", 1, "release:1", 2, "release:2"]
