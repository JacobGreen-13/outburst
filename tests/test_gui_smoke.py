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
    assert window.controls.model_id() == "yolo11n"
    assert window.controls.model_ids() == ("yolo11n", "yolo11s", "yolov8n", "yolov8s")
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


def test_single_frame_bursts_stay_hidden_until_requested(qt_app, tmp_path: Path) -> None:
    from datetime import datetime, timedelta

    from PySide6.QtCore import QSettings

    from outburst.gui.main_window import MainWindow
    from outburst.pipeline import analyze_folder
    from tests.helpers import sharp_frame, write_jpeg

    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(settings_dir))
    folder = tmp_path / "photos"
    folder.mkdir()
    base = datetime(2024, 6, 1, 12, 0, 0)
    write_jpeg(folder / "a.jpg", sharp_frame(), base)
    write_jpeg(folder / "b.jpg", sharp_frame(seed=2), base + timedelta(milliseconds=200))
    write_jpeg(folder / "c.jpg", sharp_frame(seed=3), base + timedelta(seconds=5))
    library = analyze_folder(folder, cache_dir=tmp_path / "cache", threshold=1.0)
    assert [burst.count for burst in library.bursts] == [2, 1]

    window = MainWindow(cache_dir=tmp_path / "cache")
    window.show()
    window.present(library)
    qt_app.processEvents()
    assert window.controls.show_singles() is False
    assert window.burst_list.count() == 1
    assert "1 single frame hidden" in window.statusBar().currentMessage()
    assert "a.jpg" in window.viewer.info.text()

    window.controls._singles.setChecked(True)
    qt_app.processEvents()
    assert window.burst_list.count() == 2
    assert "hidden" not in window.statusBar().currentMessage()

    window.burst_list.setCurrentRow(1)
    qt_app.processEvents()
    assert "c.jpg" in window.viewer.info.text()

    window.controls._singles.setChecked(False)
    qt_app.processEvents()
    assert window.burst_list.count() == 1
    assert "a.jpg" in window.viewer.info.text()

    solo = tmp_path / "solo"
    solo.mkdir()
    write_jpeg(solo / "only.jpg", sharp_frame(seed=4), base)
    window.present(analyze_folder(solo, cache_dir=tmp_path / "cache2", threshold=1.0))
    qt_app.processEvents()
    assert window.burst_list.count() == 0
    assert "Show single-frame bursts" in window.viewer.preview.text()
    window.close()
    qt_app.processEvents()


def test_headless_subject_toggle_does_not_download(qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PySide6.QtCore import QSettings

    from outburst.gui.main_window import MainWindow

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("headless mode must not download a model")

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", fail)
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(settings_dir))
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.controls._subject.setChecked(True)
    qt_app.processEvents()
    assert window.controls.prefer_subject() is False
    window.close()


def test_version_flag() -> None:
    from outburst.main import main

    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
