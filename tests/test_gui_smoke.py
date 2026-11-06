"""The review window can load a folder and move between frames.

``qt_app`` and ``_isolate_settings`` start offscreen Qt with a private settings file. The tests cover scanning, minimum length, minimum score, automatic keepers, rescan, a blank preview while scoring, model download, installing Ultralytics before subject or bird eye detection, zoom, filmstrip order, compare, subject box and eye circle visibility, stars, bird eye detection, a hidden single-frame folder that still reports detected eyes, and a save that stays disabled until the copy finishes.
"""

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


def test_old_show_singles_setting_lowers_the_minimum(qt_app, tmp_path: Path) -> None:
    from PySide6.QtCore import QSettings

    from outburst.gui.main_window import MainWindow

    _isolate_settings(tmp_path / "settings")
    settings = QSettings("Outburst", "Outburst")
    settings.setValue("show_single_bursts", True)
    window = MainWindow(cache_dir=tmp_path / "cache")
    qt_app.processEvents()
    assert window.controls.minimum_frames() == 1
    window.close()
    qt_app.processEvents()


def test_bursts_shorter_than_the_minimum_stay_hidden(qt_app, tmp_path: Path) -> None:
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
    assert window.controls.minimum_frames() == 2
    assert window.burst_list.count() == 1
    assert "1 burst under 2 frames hidden" in window.statusBar().currentMessage()
    assert "a.jpg" in window.viewer.info.text()

    window.controls._minimum.setValue(1)
    qt_app.processEvents()
    assert window.burst_list.count() == 2
    assert "hidden" not in window.statusBar().currentMessage()

    window.burst_list.setCurrentRow(1)
    qt_app.processEvents()
    assert "c.jpg" in window.viewer.info.text()

    window.controls._minimum.setValue(2)
    qt_app.processEvents()
    assert window.burst_list.count() == 1
    assert "a.jpg" in window.viewer.info.text()

    window.controls._minimum.setValue(3)
    qt_app.processEvents()
    assert window.burst_list.count() == 0
    assert "fewer than 3 frames" in window.viewer.preview.text()

    solo = tmp_path / "solo"
    solo.mkdir()
    write_jpeg(solo / "only.jpg", sharp_frame(seed=4), base)
    window.present(analyze_folder(solo, cache_dir=tmp_path / "cache2", threshold=1.0))
    qt_app.processEvents()
    assert window.controls.minimum_frames() == 3
    assert window.burst_list.count() == 0
    assert "Minimum frames" in window.viewer.preview.text()
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
    assert window.report_action.text() == "Save manifest…"
    assert window.report_action.iconText() == "Save manifest"
    assert "outburst-report.json" in window.report_action.toolTip()
    assert "keeper marks" in window.report_action.toolTip()
    assert window.clear_cache_action.text() == "Clear cache…"
    burst = library.bursts[0]
    recommended = burst.frames[recommended_offset(burst.frames, {item.frame.path: item for item in library.analyses})]
    assert recommended.path == written["sharp"]
    assert window._decisions[written["sharp"]] == "keep"
    assert window._origins[written["sharp"]] == "auto"
    assert window._decisions[written["second_sharp"]] == "keep"
    assert window._origins[written["second_sharp"]] == "auto"
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
    assert window._origins[written["blur"]] == "user"
    assert window.viewer.keeper_button.isChecked()
    window.close()
    qt_app.processEvents()


def test_burst_gap_replaces_automatic_keepers_and_keeps_user_marks(qt_app, tmp_path: Path) -> None:
    from outburst.cache import Cache
    from outburst.gui.main_window import MainWindow
    from outburst.output import choose_keepers
    from outburst.pipeline import analyze_folder
    from outburst.scoring import recommended_offset

    _isolate_settings(tmp_path / "settings")
    written = write_burst_folder(tmp_path / "photos")
    cache_dir = tmp_path / "cache"
    library = analyze_folder(written["folder"], cache_dir=cache_dir, threshold=1.0)
    window = MainWindow(cache_dir=cache_dir)
    window.resize(1100, 760)
    window.show()
    window.present(library)
    qt_app.processEvents()

    assert window._decisions[written["sharp"]] == "keep"
    assert window._origins[written["sharp"]] == "auto"
    assert window._decisions[written["second_sharp"]] == "keep"
    assert window._origins[written["second_sharp"]] == "auto"
    decisions = dict(window._decisions)
    origins = dict(window._origins)
    window._apply_weights()
    assert window._decisions == decisions
    assert window._origins == origins

    window._toggle_decision("reject")
    window._select_frame(1)
    window._toggle_decision("keep")
    qt_app.processEvents()
    assert window._decisions[written["sharp"]] == "reject"
    assert window._decisions[written["blur"]] == "keep"
    assert window._origins[written["blur"]] == "user"

    window.controls._threshold.setValue(10.0)
    qt_app.processEvents()
    assert window._library is not None
    assert len(window._library.bursts) == 1
    burst = window._library.bursts[0]
    best = burst.frames[recommended_offset(burst.frames, window._by_path)]
    assert window._decisions[written["sharp"]] == "reject"
    assert window._decisions[written["blur"]] == "keep"
    assert window._origins[written["blur"]] == "user"
    if best.path == written["blur"]:
        assert window._origins[best.path] == "user"
    elif window._decisions.get(best.path) == "keep":
        assert window._origins[best.path] == "auto"
    else:
        assert window._decisions[best.path] == "reject"
    for path, origin in window._origins.items():
        if origin == "auto":
            assert path == best.path
            assert window._decisions[path] == "keep"

    saved = [choice.path for choice in choose_keepers(window._library.bursts, window._by_path, window._decisions)]
    assert written["blur"] in saved
    assert written["sharp"] not in saved
    with Cache(cache_dir) as cache:
        stored = cache.decision_origins([written["blur"], written["sharp"]])
    assert stored[written["blur"]] == "user"
    assert stored[written["sharp"]] == "user"
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
    assert window.filmstrip._cards
    assert not window.viewer.preview.text().startswith("Scanning")
    window.close()
    qt_app.processEvents()


def test_rescan_blanks_the_preview_and_filmstrip_until_scoring_finishes(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from outburst.gui.main_window import MainWindow
    from outburst.gui.worker import AnalyzeWorker
    from outburst.pipeline import analyze_folder

    _isolate_settings(tmp_path / "settings")
    written = write_burst_folder(tmp_path / "photos")
    library = analyze_folder(written["folder"], cache_dir=tmp_path / "cache", threshold=1.0)
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(1100, 760)
    window.show()
    window.present(library)
    qt_app.processEvents()
    assert window.filmstrip._cards
    assert window.burst_list.count() > 0

    def spy_start(self: AnalyzeWorker) -> None:
        del self

    monkeypatch.setattr(AnalyzeWorker, "start", spy_start)
    window.rescan()
    qt_app.processEvents()

    assert window.is_scanning is True
    assert window.filmstrip._cards == []
    assert window.burst_list.count() == 0
    assert window.viewer.keeper_button.isEnabled() is False
    assert window.viewer.preview.text().startswith("Scanning and scoring")
    QTest.keyClick(window, Qt.Key.Key_Right)
    QTest.keyClick(window, Qt.Key.Key_A)
    qt_app.processEvents()
    assert window.filmstrip._cards == []
    assert window.viewer.preview.text().startswith("Scanning and scoring")
    window._scanning = False
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
    window.controls._eye.setChecked(True)
    qt_app.processEvents()
    assert window.controls.eye_scoring() is False
    assert "Bird eye model" in window.statusBar().currentMessage()
    window.close()


def test_a_missing_subject_runtime_installs_before_the_scan(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtCore import QThread

    from outburst.gui.main_window import MainWindow
    from outburst.gui.worker import AnalyzeWorker, RuntimeInstallWorker

    monkeypatch.delenv("OUTBURST_HEADLESS", raising=False)
    monkeypatch.setattr("outburst.gui.main_window.subject_runtime_ready", lambda: False)
    installs: list[str] = []
    scans: list[str] = []

    class Spy(RuntimeInstallWorker):
        def start(self, priority: QThread.Priority = QThread.Priority.InheritPriority) -> None:
            installs.append("runtime")

    monkeypatch.setattr("outburst.gui.main_window.RuntimeInstallWorker", Spy)
    monkeypatch.setattr(AnalyzeWorker, "start", lambda *_args, **_kwargs: scans.append("scan"))
    _isolate_settings(tmp_path / "settings")
    window = MainWindow(cache_dir=tmp_path / "cache")
    window._folder = tmp_path / "photos"
    window.controls.set_eye_scoring(True)
    window.controls.set_subject_checked(True)
    window._start_worker()
    qt_app.processEvents()
    assert installs == ["runtime"]
    assert scans == []
    assert "PyTorch" in window.statusBar().currentMessage()
    window.close()
    qt_app.processEvents()


def test_headless_missing_runtime_still_scans(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from outburst.gui.main_window import MainWindow
    from outburst.gui.worker import AnalyzeWorker, RuntimeInstallWorker

    installs: list[str] = []
    scans: list[str] = []
    monkeypatch.setattr("outburst.gui.main_window.subject_runtime_ready", lambda: False)
    monkeypatch.setattr(MainWindow, "_model_ready", lambda _self: True)
    monkeypatch.setattr(MainWindow, "_bird_eye_ready", lambda _self: True)
    monkeypatch.setattr(
        RuntimeInstallWorker,
        "start",
        lambda *_args, **_kwargs: installs.append("runtime"),
    )
    monkeypatch.setattr(AnalyzeWorker, "start", lambda *_args, **_kwargs: scans.append("scan"))
    _isolate_settings(tmp_path / "settings")
    window = MainWindow(cache_dir=tmp_path / "cache")
    window._folder = tmp_path / "photos"
    window.controls.set_eye_scoring(True)
    window.controls.set_subject_checked(True)
    window._start_worker()
    qt_app.processEvents()
    assert installs == []
    assert scans == ["scan"]
    window.close()
    qt_app.processEvents()


def test_a_finished_runtime_install_starts_the_scan(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from outburst.gui.main_window import MainWindow
    from outburst.gui.worker import AnalyzeWorker

    ready = {"ok": False}
    scans: list[str] = []
    monkeypatch.delenv("OUTBURST_HEADLESS", raising=False)
    monkeypatch.setattr("outburst.gui.main_window.subject_runtime_ready", lambda: ready["ok"])
    monkeypatch.setattr(MainWindow, "_model_ready", lambda _self: True)
    monkeypatch.setattr(MainWindow, "_bird_eye_ready", lambda _self: True)
    monkeypatch.setattr(AnalyzeWorker, "start", lambda *_args, **_kwargs: scans.append("scan"))
    _isolate_settings(tmp_path / "settings")
    window = MainWindow(cache_dir=tmp_path / "cache")
    window._folder = tmp_path / "photos"
    window.controls.set_eye_scoring(True)
    window._download_then_scan = True
    window._downloading = True
    ready["ok"] = True
    window._on_runtime_installed()
    qt_app.processEvents()
    assert scans == ["scan"]
    assert window.controls.eye_scoring() is True
    window.close()
    qt_app.processEvents()


def test_a_failed_runtime_install_leaves_detection_off(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from outburst.gui.main_window import MainWindow
    from outburst.gui.worker import AnalyzeWorker

    notes: list[str] = []
    scans: list[str] = []
    monkeypatch.delenv("OUTBURST_HEADLESS", raising=False)
    monkeypatch.setattr(
        "outburst.gui.main_window.notify",
        lambda *_args, **_kwargs: notes.append(str(_args[1])),
    )
    monkeypatch.setattr(AnalyzeWorker, "start", lambda *_args, **_kwargs: scans.append("scan"))
    _isolate_settings(tmp_path / "settings")
    window = MainWindow(cache_dir=tmp_path / "cache")
    window._folder = tmp_path / "photos"
    window.controls.set_eye_scoring(True)
    window.controls.set_subject_checked(True)
    window._downloading = True
    window._download_then_scan = True
    window._on_runtime_install_failed("no network")
    qt_app.processEvents()
    assert window.controls.eye_scoring() is False
    assert window.controls.prefer_subject() is False
    assert notes == ["Subject detection"]
    assert scans == ["scan"]
    window.close()
    qt_app.processEvents()


def test_zoom_steps() -> None:
    from outburst.gui.image_viewer import next_zoom

    assert next_zoom(None, 1) == 1.0
    assert next_zoom(1.0, 1) == 2.0
    assert next_zoom(2.0, 1) == 4.0
    assert next_zoom(4.0, 1) == 4.0
    assert next_zoom(4.0, -1) == 2.0
    assert next_zoom(1.0, -1) is None
    assert next_zoom(None, -1) is None


def test_filmstrip_sort_follows_time_or_score(qt_app, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from outburst.display import filmstrip_order
    from outburst.gui.main_window import MainWindow
    from outburst.pipeline import analyze_folder

    _isolate_settings(tmp_path / "settings")
    written = write_burst_folder(tmp_path / "photos")
    library = analyze_folder(written["folder"], cache_dir=tmp_path / "cache", threshold=1.0)
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(1100, 760)
    window.show()
    window.present(library)
    qt_app.processEvents()
    window.burst_list.setCurrentRow(1)
    qt_app.processEvents()

    burst = window._current_burst()
    assert burst is not None and burst.count == 2
    assert [card.index for card in window.filmstrip._cards] == [0, 1]
    held = window.current_frame_index
    QTest.mouseClick(window.sort_score, Qt.MouseButton.LeftButton)
    qt_app.processEvents()
    scores = [
        None if (analysis := window._by_path.get(frame.path)) is None else analysis.composite
        for frame in burst.frames
    ]
    expected = filmstrip_order(scores, "score")
    assert expected == [1, 0]
    assert [card.index for card in window.filmstrip._cards] == expected
    assert window.current_frame_index == held
    QTest.keyClick(window, Qt.Key.Key_Home)
    QTest.keyClick(window, Qt.Key.Key_Right)
    qt_app.processEvents()
    assert window.current_frame_index == expected[1]
    QTest.keyClick(window, Qt.Key.Key_End)
    qt_app.processEvents()
    assert window.current_frame_index == expected[-1]
    window.close()
    qt_app.processEvents()

    restored = MainWindow(cache_dir=tmp_path / "cache")
    assert restored.sort_score.isChecked()
    restored.close()
    qt_app.processEvents()


def test_compare_loupe_and_obvious_misses(qt_app, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from outburst.display import obvious_misses
    from outburst.gui.main_window import MainWindow
    from outburst.pipeline import analyze_folder
    from outburst.scoring import compare_partner_offset

    _isolate_settings(tmp_path / "settings")
    written = write_burst_folder(tmp_path / "photos")
    library = analyze_folder(written["folder"], cache_dir=tmp_path / "cache", threshold=1.0)
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(1100, 760)
    window.show()
    window.present(library)
    qt_app.processEvents()

    burst = window._current_burst()
    assert burst is not None
    partner = compare_partner_offset(burst.frames, window._by_path, window.current_frame_index)
    assert partner is not None
    QTest.keyClick(window, Qt.Key.Key_C)
    qt_app.processEvents()
    assert window.viewer.partner_box.isVisible()
    assert "Next sharpest" in window.viewer.partner_caption.text()
    QTest.mouseClick(window.viewer.partner, Qt.MouseButton.LeftButton)
    qt_app.processEvents()
    assert window.current_frame_index == partner
    assert "Best in the burst" in window.viewer.partner_caption.text()

    QTest.keyClick(window, Qt.Key.Key_Z)
    QTest.keyClick(window, Qt.Key.Key_F)
    qt_app.processEvents()
    assert window._loupe is True
    assert window._peaking is True
    assert window.viewer.hint.isVisible()
    assert "1:1" in window.viewer.hint.text()
    assert "Green" in window.viewer.hint.text()

    QTest.keyClick(window, Qt.Key.Key_Plus)
    qt_app.processEvents()
    assert window.viewer.preview.scale() == 2
    assert "2:1" in window.viewer.hint.text()
    QTest.keyClick(window, Qt.Key.Key_Minus)
    qt_app.processEvents()
    assert window.viewer.preview.scale() == 1

    QTest.keyClick(window, Qt.Key.Key_BracketRight)
    qt_app.processEvents()
    caption = window.viewer.partner_caption.text()
    assert caption.startswith("Frame ")
    assert "Best in the burst" not in caption
    burst = window._current_burst()
    assert burst is not None and burst.count >= 3
    target = next(
        index
        for index in range(burst.count)
        if index != window.current_frame_index and index != window._compare_index
    )
    QTest.mouseClick(
        window.filmstrip._cards[target],
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.ShiftModifier,
    )
    qt_app.processEvents()
    assert window._compare_index == target
    assert window.viewer.partner_box.isVisible()

    window._select_frame(0)
    qt_app.processEvents()
    burst = window._current_burst()
    assert burst is not None
    misses = obvious_misses(burst.frames, window._by_path, window._decisions)
    assert misses
    assert written["sharp"] not in misses
    QTest.keyClick(window, Qt.Key.Key_G)
    qt_app.processEvents()
    for path in misses:
        assert window._decisions[path] == "reject"
    assert window._decisions[written["sharp"]] == "keep"
    QTest.keyClick(window, Qt.Key.Key_G)
    qt_app.processEvents()
    for path in misses:
        assert path not in window._decisions
    assert window._decisions[written["sharp"]] == "keep"
    window.close()
    qt_app.processEvents()


def test_star_keys_are_saved_apart_from_the_pick_flag(qt_app, tmp_path: Path) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from outburst.cache import Cache
    from outburst.gui.main_window import MainWindow
    from outburst.pipeline import analyze_folder

    _isolate_settings(tmp_path / "settings")
    written = write_burst_folder(tmp_path / "photos")
    cache_dir = tmp_path / "cache"
    library = analyze_folder(written["folder"], cache_dir=cache_dir, threshold=1.0)
    window = MainWindow(cache_dir=cache_dir)
    window.resize(1100, 760)
    window.show()
    window.present(library)
    qt_app.processEvents()

    assert window._decisions[written["sharp"]] == "keep"
    QTest.keyClick(window, Qt.Key.Key_4)
    qt_app.processEvents()
    assert window._stars[written["sharp"]] == 4
    assert window._decisions[written["sharp"]] == "keep"
    assert "★★★★☆" in window.viewer.info.text()
    assert window.viewer.star_buttons[3].property("filled") == "true"
    assert window.viewer.star_buttons[4].property("filled") == "false"
    QTest.keyClick(window, Qt.Key.Key_4)
    qt_app.processEvents()
    assert written["sharp"] not in window._stars
    with Cache(cache_dir) as cache:
        cache.set_stars(written["blur"], 2)
    window.present(library)
    qt_app.processEvents()
    assert window._stars[written["blur"]] == 2
    window.close()
    qt_app.processEvents()


def test_eye_scoring_marks_the_sharpest_eye_and_can_be_toggled(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta

    from outburst.gui.main_window import MainWindow
    from outburst.gui.worker import AnalyzeWorker
    from outburst.pipeline import analyze_folder
    from tests.helpers import write_jpeg
    from tests.test_eyes import PortraitEyeFinder, _portrait

    _isolate_settings(tmp_path / "settings")
    monkeypatch.setenv("OUTBURST_MODELS", str(tmp_path / "models"))
    _ready_weights(tmp_path / "models" / "eye_pose_v0.pt")
    folder = tmp_path / "photos"
    folder.mkdir()
    moment = datetime(2024, 7, 2, 9, 0, 0)
    sharp = folder / "sharp.jpg"
    soft = folder / "soft.jpg"
    write_jpeg(sharp, _portrait(36), moment)
    write_jpeg(soft, _portrait(36, blur=21), moment + timedelta(milliseconds=200))
    library = analyze_folder(
        folder,
        cache_dir=tmp_path / "cache",
        threshold=1.0,
        eye_scoring=True,
        eye_finder=PortraitEyeFinder(),
    )
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(1100, 760)
    window.show()
    assert window.controls.eye_scoring() is False
    assert window.controls._eye.text() == "Bird eye detection"
    window.present(library)
    qt_app.processEvents()

    assert window._decisions[sharp] == "keep"
    assert window._origins[sharp] == "auto"
    assert soft not in window._decisions
    assert "Bird eye detected" in window.viewer.detail.text()
    assert "Bird eye detected" not in window.viewer.info.text()
    assert window.viewer.detail.isHidden()
    assert "2 eyes" in window.burst_list.item(0).text()
    assert "bird eye detection" in window.statusBar().currentMessage()
    assert "2 bird eyes" in window.statusBar().currentMessage()
    assert any("EYE" in card.score_text for card in window._thumbs())
    assert window.viewer.keeper_button.isChecked()

    window._apply_decision(soft, "keep", origin="user")
    window._mark_recommended_keepers()
    assert window._decisions[sharp] == "keep"
    assert window._origins[sharp] == "auto"
    assert window._decisions[soft] == "keep"
    assert window._origins[soft] == "user"

    window._apply_decision(sharp, "reject", origin="user")
    window._apply_decision(soft, "keep", origin="auto")
    window._mark_recommended_keepers()
    assert window._decisions[sharp] == "reject"
    assert soft not in window._decisions

    started: list[tuple[bool, str | None]] = []

    def spy_start(self: AnalyzeWorker) -> None:
        model = self._eye_model.name if self._eye_model is not None else None
        started.append((self._eye_scoring, model))

    monkeypatch.setattr(AnalyzeWorker, "start", spy_start)
    window._apply_decision(soft, "keep", origin="user")
    window._apply_decision(sharp, "keep", origin="auto")
    window.controls._eye.setChecked(True)
    qt_app.processEvents()
    assert started == [(True, "eye_pose_v0.pt")]
    assert sharp not in window._decisions
    assert window._decisions[soft] == "keep"
    assert window._origins[soft] == "user"
    window._scanning = False
    window.close()
    qt_app.processEvents()


def test_hidden_single_frames_still_say_when_an_eye_was_found(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime, timedelta

    from outburst.gui.main_window import MainWindow
    from outburst.pipeline import analyze_folder
    from tests.helpers import write_jpeg
    from tests.test_eyes import PortraitEyeFinder, _portrait

    _isolate_settings(tmp_path / "settings")
    monkeypatch.setenv("OUTBURST_MODELS", str(tmp_path / "models"))
    folder = tmp_path / "photos"
    folder.mkdir()
    moment = datetime(2026, 6, 8, 7, 42, 0)
    write_jpeg(folder / "heron.jpg", _portrait(36), moment)
    write_jpeg(folder / "duck.jpg", _portrait(36, seed=2), moment + timedelta(seconds=62))
    library = analyze_folder(
        folder,
        cache_dir=tmp_path / "cache",
        threshold=1.0,
        eye_scoring=True,
        eye_finder=PortraitEyeFinder(),
    )
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.show()
    window.present(library)
    qt_app.processEvents()
    assert window.controls.minimum_frames() == 2
    assert window.burst_list.count() == 0
    assert "found an eye on 2 frames" in window.viewer.preview.text()
    assert "2 bird eyes" in window.statusBar().currentMessage()
    window.close()
    qt_app.processEvents()


def test_save_keepers_disables_save_until_the_copy_finishes(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading
    import time

    from outburst.gui.dialogs import ExportRequest
    from outburst.gui.main_window import MainWindow
    from outburst.output import export_keepers
    from outburst.pipeline import analyze_folder

    _isolate_settings(tmp_path / "settings")
    written = write_burst_folder(tmp_path / "photos")
    library = analyze_folder(written["folder"], cache_dir=tmp_path / "cache", threshold=1.0)
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(1100, 760)
    window.show()
    window.present(library)
    qt_app.processEvents()
    destination = tmp_path / "keepers"
    monkeypatch.setattr(
        "outburst.gui.main_window.ExportKeepersDialog.ask",
        lambda *_args, **_kwargs: ExportRequest(destination, "copy"),
    )
    started = threading.Event()
    release = threading.Event()

    def wrapped(*args, **kwargs):  # type: ignore[no-untyped-def]
        started.set()
        assert release.wait(5)
        return export_keepers(*args, **kwargs)

    monkeypatch.setattr("outburst.gui.worker.export_keepers", wrapped)
    window._export_keepers()
    assert started.wait(3)
    qt_app.processEvents()
    assert window.export_action.isEnabled() is False
    assert window.xmp_action.isEnabled() is False
    assert window.progress.isVisible()
    release.set()
    for _ in range(100):
        qt_app.processEvents()
        if not window._saving:
            break
        time.sleep(0.02)
    assert window._saving is False
    assert window.export_action.isEnabled() is True
    assert (destination / written["sharp"].name).is_file()
    assert (destination / written["second_sharp"].name).is_file()
    window.close()
    qt_app.processEvents()


def test_minimum_score_hides_weaker_frames_and_keeps_marks(qt_app, tmp_path: Path) -> None:
    from datetime import datetime

    from outburst.gui.main_window import MainWindow
    from outburst.models import Burst, Frame, FrameAnalysis, Library, Metrics, Weights
    from outburst.output import choose_keepers

    _isolate_settings(tmp_path / "settings")
    when = datetime(2024, 6, 1, 12, 0, 0)
    folder = tmp_path / "photos"
    folder.mkdir()

    def frame(name: str, score: float) -> tuple[Frame, FrameAnalysis]:
        path = folder / name
        item = Frame(path, "jpeg", when, "exif", None, 1, 1)
        analysis = FrameAnalysis(item, Metrics(1, 1, 1, 0.5, 0.0, 0.0), score, None, None, None)
        return item, analysis

    strong, strong_analysis = frame("strong.jpg", 0.80)
    mid, mid_analysis = frame("mid.jpg", 0.40)
    low, low_analysis = frame("low.jpg", 0.20)
    soft, soft_analysis = frame("soft-a.jpg", 0.30)
    softer, softer_analysis = frame("soft-b.jpg", 0.10)
    library = Library(
        folder,
        (strong_analysis, mid_analysis, low_analysis, soft_analysis, softer_analysis),
        (
            Burst("burst-001", (strong, mid, low)),
            Burst("burst-002", (soft, softer)),
        ),
        (),
        1.0,
        Weights(),
    )
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(1100, 760)
    window.show()
    window.present(library)
    qt_app.processEvents()
    assert window.burst_list.count() == 2
    assert len(window.filmstrip._cards) == 3
    assert len(choose_keepers(library.bursts, window._by_path, window._decisions, minimum_frames=2)) == 2

    window.controls.set_minimum_score(50)
    window.controls._score_filter.setChecked(True)
    qt_app.processEvents()
    assert window.controls.score_floor() == 50
    assert window.burst_list.count() == 1
    assert "1 of 3 frames" in window.burst_list.item(0).text()
    assert [card.index for card in window.filmstrip._cards] == [0]
    assert "under 50" in window.statusBar().currentMessage()
    window._step_frame(1)
    qt_app.processEvents()
    assert window.current_frame_index == 0
    assert len(choose_keepers(library.bursts, window._by_path, window._decisions, minimum_frames=2)) == 2

    window.controls._score_minimum.setValue(90)
    qt_app.processEvents()
    assert window.burst_list.count() == 0
    assert "Minimum score" in window.viewer.preview.text()

    window.controls._score_filter.setChecked(False)
    qt_app.processEvents()
    assert window.burst_list.count() == 2
    assert len(window.filmstrip._cards) == 3
    window.close()
    qt_app.processEvents()


def test_subject_box_and_eye_circle_can_be_hidden(
    qt_app, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import datetime

    from PySide6.QtGui import QColor, QImage

    from outburst.gui.main_window import MainWindow
    from outburst.gui.worker import AnalyzeWorker
    from outburst.models import Burst, Frame, FrameAnalysis, Library, Metrics, Weights

    _isolate_settings(tmp_path / "settings")
    preview = tmp_path / "bird.jpg"
    image = QImage(320, 240, QImage.Format.Format_RGB32)
    image.fill(QColor("#336699"))
    assert image.save(str(preview))
    when = datetime(2024, 6, 1, 12, 0, 0)
    metrics = Metrics(
        10,
        12,
        3,
        0.8,
        0.0,
        0.0,
        subject_box=(40, 30, 80, 60),
        eye_center=(70, 55),
        eye_radius=12,
    )

    def frame(name: str) -> tuple[Frame, FrameAnalysis]:
        path = tmp_path / name
        path.write_bytes(preview.read_bytes())
        item = Frame(path, "jpeg", when, "exif", None, 1, 1)
        return item, FrameAnalysis(item, metrics, 0.8, path, path, None)

    first, first_analysis = frame("one.jpg")
    second, second_analysis = frame("two.jpg")
    library = Library(
        tmp_path,
        (first_analysis, second_analysis),
        (Burst("burst-001", (first, second)),),
        (),
        1.0,
        Weights(prefer_subject=True),
        subject_active=True,
        eye_active=True,
    )
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(1100, 760)
    window.show()
    scans: list[str] = []
    monkeypatch.setattr(AnalyzeWorker, "start", lambda *_args, **_kwargs: scans.append("scan"))
    window.present(library)
    qt_app.processEvents()
    subject_on = window.controls.prefer_subject()
    eye_on = window.controls.eye_scoring()
    assert window.controls.show_subject_box() is True
    assert window.controls.show_eye_circle() is True
    assert window.viewer.preview._box == (40, 30, 80, 60)
    assert window.viewer.preview._circle == (70, 55, 12)
    assert "Bird eye detected" in window.viewer.detail.text()
    assert "Bird eye detected" not in window.viewer.info.text()
    assert window.viewer.detail.isHidden()

    window.controls._show_box.setChecked(False)
    qt_app.processEvents()
    assert window.controls.prefer_subject() is subject_on
    assert window.viewer.preview._box is None
    assert window.viewer.preview._circle == (70, 55, 12)
    assert window.viewer.preview._focus == (70, 55)
    assert scans == []

    window.controls._show_eye.setChecked(False)
    qt_app.processEvents()
    assert window.controls.eye_scoring() is eye_on
    assert window.viewer.preview._circle is None
    assert window.viewer.preview._focus == (70, 55)
    assert "Bird eye detected" in window.viewer.detail.text()
    assert "Bird eye detected" not in window.viewer.info.text()
    assert window.viewer.detail.isHidden()
    assert scans == []

    window.compare_action.setChecked(True)
    qt_app.processEvents()
    assert window.viewer.partner._box is None
    assert window.viewer.partner._circle is None
    assert window.viewer.partner._focus == (70, 55)

    window.controls._show_box.setChecked(True)
    window.controls._show_eye.setChecked(True)
    qt_app.processEvents()
    assert window.viewer.partner._box == (40, 30, 80, 60)
    assert window.viewer.partner._circle == (70, 55, 12)
    assert scans == []

    window.controls._show_box.setChecked(False)
    window._save_settings()
    window.close()
    qt_app.processEvents()
    restored = MainWindow(cache_dir=tmp_path / "cache")
    assert restored.controls.show_subject_box() is False
    assert restored.controls.show_eye_circle() is True
    assert restored.controls.prefer_subject() is False
    assert restored.controls.eye_scoring() is False
    restored.close()
    qt_app.processEvents()


def test_info_button_hides_the_lines_after_the_laplacian(qt_app, tmp_path: Path) -> None:
    from datetime import datetime

    from PySide6.QtGui import QColor, QImage

    from outburst.gui.main_window import MainWindow
    from outburst.models import Burst, Frame, FrameAnalysis, Library, Metrics, Weights

    _isolate_settings(tmp_path / "settings")
    preview = tmp_path / "bird.jpg"
    image = QImage(160, 120, QImage.Format.Format_RGB32)
    image.fill(QColor("#336699"))
    assert image.save(str(preview))
    frame = Frame(preview, "jpeg", datetime(2024, 6, 1, 12, 0, 0), "exif", "Sample", 1, 1)
    metrics = Metrics(45, 119, 209, 0.8, 0.0, 0.0, critical_sharpness=123)
    analysis = FrameAnalysis(frame, metrics, 0.79, preview, preview, None)
    library = Library(tmp_path, (analysis,), (Burst("burst-001", (frame,)),), (), 1.0, Weights())
    window = MainWindow(cache_dir=tmp_path / "cache")
    window.resize(900, 640)
    window.show()
    window.controls.set_minimum_frames(1)
    window.present(library)
    qt_app.processEvents()
    assert window.viewer.info_button.text() == "Info"
    assert window.viewer.info_open() is False
    assert window.viewer.detail.isHidden()
    assert "Laplacian" in window.viewer.info.text()
    assert "Frame 1 of 1" not in window.viewer.info.text()
    assert "Full-frame sharpness" in window.viewer.detail.text()
    assert "Exposure looks balanced" in window.viewer.detail.text()

    window.viewer.info_button.setChecked(True)
    qt_app.processEvents()
    assert window.viewer.detail.isVisible()
    window.viewer.info_button.setChecked(False)
    qt_app.processEvents()
    assert window.viewer.detail.isHidden()

    window._save_settings()
    window.close()
    qt_app.processEvents()
    restored = MainWindow(cache_dir=tmp_path / "cache")
    assert restored.viewer.info_open() is False
    restored.viewer.set_info_open(True)
    restored._save_settings()
    restored.close()
    qt_app.processEvents()
    again = MainWindow(cache_dir=tmp_path / "cache")
    assert again.viewer.info_open() is True
    again.close()
    qt_app.processEvents()


def test_version_flag() -> None:
    from outburst.main import main

    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0
