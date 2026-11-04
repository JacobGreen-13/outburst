"""The review window can load a folder and move between frames."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.helpers import write_burst_folder


@pytest.fixture
def qt_app():
    pytest.importorskip("PySide6")
    try:
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on the host's display libraries
        pytest.skip(f"Qt display libraries are unavailable: {exc}")
    application = QApplication.instance() or QApplication([])
    yield application


def test_window_reviews_a_burst(qt_app, tmp_path: Path) -> None:
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtTest import QTest

    from outburst.gui.main_window import MainWindow

    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(settings_dir))
    written = write_burst_folder(tmp_path / "photos")
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(1100, 760)
    window.show()
    qt_app.processEvents()
    window.open_path(written["folder"])
    deadline = time.time() + 30
    while time.time() < deadline and (window.library is None or window.is_scanning):
        qt_app.processEvents()
        time.sleep(0.02)
    assert window.library is not None
    assert window.is_scanning is False
    assert len(window.library.bursts) == 2
    assert window.windowTitle() == "Outburst"
    assert "RECOMMENDED" in window.viewer.info.text()
    assert written["sharp"].name in window.viewer.info.text()

    before = window.current_frame_index
    QTest.keyClick(window, Qt.Key.Key_Right)
    qt_app.processEvents()
    assert window.current_frame_index == before + 1

    QTest.keyClick(window, Qt.Key.Key_Down)
    qt_app.processEvents()
    assert window.current_burst_index == 1
    window.close()
    qt_app.processEvents()


def test_version_flag() -> None:
    from outburst.main import main

    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
