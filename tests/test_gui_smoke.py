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


def _isolate_settings(directory: Path) -> None:
    from PySide6.QtCore import QSettings

    directory.mkdir(parents=True, exist_ok=True)
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for fmt in (QSettings.Format.IniFormat, QSettings.Format.NativeFormat):
        QSettings.setPath(fmt, QSettings.Scope.UserScope, str(directory))


def test_window_reviews_a_burst(qt_app, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from outburst.gui.main_window import MainWindow

    _isolate_settings(tmp_path / "settings")
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

    from outburst.gui.main_window import MainWindow
    from outburst.pipeline import analyze_folder
    from tests.helpers import sharp_frame, write_jpeg

    _isolate_settings(tmp_path / "settings")
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


def test_recommended_frame_starts_marked_and_save_uses_that_mark(qt_app, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from outburst.gui.main_window import MainWindow
    from outburst.output import choose_keepers
    from outburst.pipeline import analyze_folder
    from outburst.scoring import recommended_offset

    _isolate_settings(tmp_path / "settings")
    written = write_burst_folder(tmp_path / "photos")
    library = analyze_folder(written["folder"], cache_dir=tmp_path / "cache", threshold=1.0)
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(1100, 760)
    window.show()
    window.present(library)
    qt_app.processEvents()

    assert window.export_action.text().startswith("Save keepers")
    assert window.clear_cache_action.text() == "Clear cache…"
    burst = library.bursts[0]
    recommended = burst.frames[recommended_offset(burst.frames, {item.frame.path: item for item in library.analyses})]
    assert recommended.path == written["sharp"]
    assert window._decisions[written["sharp"]] == "keep"
    assert window._decisions[written["second_sharp"]] == "keep"
    assert window.viewer.keeper_button.isChecked()
    assert window.viewer.keeper_button.text() == "Marked keeper"
    assert "KEEPER" in window.viewer.info.text()

    window.viewer.keeper_button.click()
    qt_app.processEvents()
    assert written["sharp"] not in window._decisions
    assert window.viewer.keeper_button.isChecked() is False
    assert window.viewer.keeper_button.text() == "Mark as keeper"
    saved = [choice.path for choice in choose_keepers(library.bursts, window._by_path, window._decisions)]
    assert written["sharp"] not in saved
    assert written["second_sharp"] in saved

    QTest.keyClick(window, Qt.Key.Key_Right)
    qt_app.processEvents()
    window.viewer.keeper_button.click()
    qt_app.processEvents()
    assert window._decisions[written["blur"]] == "keep"
    assert window.viewer.keeper_button.isChecked()
    window.close()
    qt_app.processEvents()


def test_rescan_forgets_cached_measurements_for_the_open_folder(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from outburst.cache import Cache
    from outburst.gui.main_window import MainWindow
    from outburst.pipeline import analyze_folder

    _isolate_settings(tmp_path / "settings")
    written = write_burst_folder(tmp_path / "photos")
    cache_dir = tmp_path / "cache"
    library = analyze_folder(written["folder"], cache_dir=cache_dir, threshold=1.0)
    window = MainWindow(cache_dir=cache_dir)
    window.present(library)
    qt_app.processEvents()
    photo = written["sharp"]
    stat = photo.stat()
    with Cache(cache_dir) as cache:
        assert cache.get(photo, stat.st_mtime_ns, stat.st_size, "") is not None

    forgotten: list[Path] = []
    original = Cache.forget_folder

    def spy(self: Cache, folder: Path) -> int:
        forgotten.append(Path(folder))
        return original(self, folder)

    monkeypatch.setattr(Cache, "forget_folder", spy)
    window.rescan()
    deadline = time.time() + 30
    while time.time() < deadline and (window.library is None or window.is_scanning):
        qt_app.processEvents()
        time.sleep(0.02)
    assert window.is_scanning is False
    assert forgotten == [written["folder"].resolve()]
    assert window._decisions[written["sharp"]] == "keep"
    window.close()
    qt_app.processEvents()


def _ready_weights(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"PK" + b"0" * (1_000_000 - 2))


def test_choosing_a_model_turns_subject_scoring_on_and_rescans_with_it(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from outburst.gui.main_window import MainWindow
    from outburst.gui.worker import AnalyzeWorker
    from outburst.pipeline import analyze_folder

    _isolate_settings(tmp_path / "settings")
    monkeypatch.setenv("OUTBURST_MODELS", str(tmp_path / "models"))
    _ready_weights(tmp_path / "models" / "yolo11s.pt")
    written = write_burst_folder(tmp_path / "photos")
    library = analyze_folder(written["folder"], cache_dir=tmp_path / "cache", threshold=1.0)
    started: list[Path | None] = []

    def spy_start(self: AnalyzeWorker) -> None:
        started.append(self._subject_model)

    monkeypatch.setattr(AnalyzeWorker, "start", spy_start)
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.present(library)
    qt_app.processEvents()
    assert window.controls.prefer_subject() is False

    window.controls._model.setCurrentIndex(window.controls._model.findData("yolo11s"))
    qt_app.processEvents()

    assert window.controls.prefer_subject() is True
    assert window.controls.model_id() == "yolo11s"
    assert [path.name if path is not None else None for path in started] == ["yolo11s.pt"]
    window._scanning = False
    window.close()
    qt_app.processEvents()


def test_rescan_downloads_the_selected_model_before_scoring(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from outburst.gui.main_window import MainWindow
    from outburst.pipeline import analyze_folder

    _isolate_settings(tmp_path / "settings")
    monkeypatch.setenv("OUTBURST_MODELS", str(tmp_path / "models"))
    written = write_burst_folder(tmp_path / "photos")
    library = analyze_folder(written["folder"], cache_dir=tmp_path / "cache", threshold=1.0)
    calls: list[tuple[str, bool, bool]] = []

    def spy_download(self: MainWindow, then_scan: bool) -> None:
        calls.append((self.controls.model_id(), then_scan, self.controls.prefer_subject()))

    monkeypatch.setattr(MainWindow, "_begin_download", spy_download)
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.present(library)
    qt_app.processEvents()

    window.controls._model.setCurrentIndex(window.controls._model.findData("yolov8s"))
    qt_app.processEvents()
    assert calls == [("yolov8s", True, True)]

    calls.clear()
    window.controls.set_model_id("yolo11n")
    window.controls.set_subject_checked(True)
    window.rescan()
    qt_app.processEvents()
    assert calls == [("yolo11n", True, True)]
    window.close()
    qt_app.processEvents()


def test_headless_subject_toggle_does_not_download(qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from outburst.gui.main_window import MainWindow

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("headless mode must not download a model")

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", fail)
    _isolate_settings(tmp_path / "settings")
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
