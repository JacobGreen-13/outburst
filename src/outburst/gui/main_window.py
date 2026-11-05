"""The Outburst review window."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, QSize, Qt, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QSizePolicy,
    QSplitter,
    QStyle,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from outburst import __version__
from outburst.cache import Cache, default_cache_dir
from outburst.display import (
    burst_list_text,
    frame_info_html,
    obvious_misses,
    score_caption,
    thumb_badge,
    visible_bursts,
)
from outburst.gui.burst_view import Filmstrip, Thumb
from outburst.gui.controls import ControlPanel
from outburst.gui.dialogs import ExportKeepersDialog, ask, notify, show_shortcuts, show_warnings
from outburst.gui.image_viewer import ImageViewer
from outburst.gui.styles import build_app_icon
from outburst.gui.worker import AnalyzeWorker, ModelFetchWorker
from outburst.models import Burst, FrameAnalysis, Library
from outburst.output import choose_keepers, export_keepers, write_library_sidecars, write_reports
from outburst.pipeline import with_threshold, with_weights
from outburst.scoring import compare_partner_offset, eye_chooses_best, recommended_offset
from outburst.subject import (
    DEFAULT_MODEL_ID,
    MODEL_OPTIONS,
    model_is_ready,
    model_path,
    models_dir,
    option_by_id,
)

log = logging.getLogger(__name__)

_WELCOME = (
    "Open a folder of RAW bursts to start culling.\n\n"
    "Outburst groups frames shot close together, scores sharpness and exposure "
    "from embedded previews, and marks the recommended frame as a keeper.\n\n"
    "Use Mark as keeper under the preview to choose what Save keepers writes. "
    "Original files stay untouched."
)


def _short_bursts_hidden(minimum: int) -> str:
    return (
        f"Every burst in this folder has fewer than {minimum} frames.\n\n"
        "Lower Minimum frames to review them."
    )


class MainWindow(QMainWindow):
    def __init__(self, cache_dir: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Outburst")
        self.setWindowIcon(build_app_icon())
        self.resize(1280, 840)
        self.last_notice: tuple[str, str] | None = None
        self._cache_dir = cache_dir or default_cache_dir()
        self._library: Library | None = None
        self._by_path: dict[Path, FrameAnalysis] = {}
        self._decisions: dict[Path, str] = {}
        self._origins: dict[Path, str] = {}
        self._stars: dict[Path, int] = {}
        self._compare = False
        self._loupe = False
        self._peaking = False
        self._obvious_burst_id: str | None = None
        self._obvious_rejects: list[Path] = []
        self._folder: Path | None = None
        self._dialog_start = Path.home()
        self._model_path: Path | None = None
        self._download_worker: ModelFetchWorker | None = None
        self._downloading = False
        self._download_then_scan = False
        self._download_model_id: str | None = None
        self._replace_download = False
        self._burst_index = 0
        self._frame_index = 0
        self._warnings: tuple[str, ...] = ()
        self._worker: AnalyzeWorker | None = None
        self._scan_id = 0
        self._scanning = False
        self._build_ui()
        self.viewer.keeper_clicked.connect(lambda: self._toggle_decision("keep"))
        self.viewer.reject_clicked.connect(lambda: self._toggle_decision("reject"))
        self.viewer.star_clicked.connect(self._set_stars)
        self._build_actions()
        self._load_settings()
        self._sync_actions()
        application = QApplication.instance()
        if application is not None:
            application.installEventFilter(self)

    @property
    def library(self) -> Library | None:
        return self._library

    @property
    def is_scanning(self) -> bool:
        return self._scanning

    @property
    def current_burst_index(self) -> int:
        return self._burst_index

    @property
    def current_frame_index(self) -> int:
        return self._frame_index

    def present(self, library: Library) -> None:
        """Show a finished library. Tests and the scan worker both use this."""

        self._library = library
        self._folder = library.folder
        self._by_path = {item.frame.path: item for item in library.analyses}
        self._warnings = library.warnings
        self._load_decisions()
        self._mark_recommended_keepers()
        self._set_folder_label()
        self.controls.set_threshold(library.threshold)
        self._refresh_burst_list(prefer_path=None)
        self._sync_actions()
        self.statusBar().showMessage(self._status_summary())
        if library.warnings and not library.analyses:
            self._notify("Nothing to cull", library.warnings[0], "info")

    def open_path(self, folder: Path) -> None:
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            self._notify("Folder not found", f"Outburst couldn't open {folder}.", "warning")
            return
        if not self._stop_worker():
            self._notify(
                "Still working",
                "Outburst is still stopping the previous scan. Try again in a moment.",
                "warning",
            )
            return
        self._folder = folder
        self._dialog_start = folder
        self._set_folder_label()
        self._start_worker()

    def rescan(self) -> None:
        if self._folder is None:
            return
        folder = self._folder
        if not self._stop_worker():
            self._notify(
                "Still working",
                "Outburst is still stopping the previous scan. Try again in a moment.",
                "warning",
            )
            return
        self._forget_folder_cache(folder)
        if self.controls.prefer_subject() and not self._model_ready():
            self._begin_download(then_scan=True)
            return
        self.open_path(folder)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        if event.type() == QEvent.Type.KeyPress and watched is QApplication.focusWidget():
            if not _typing_target(watched) and self._handle_key(event.key()):
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if not _typing_target(QApplication.focusWidget()) and self._handle_key(event.key()):
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        self._save_settings()
        if self._download_worker is not None and self._download_worker.isRunning():
            self._download_worker.abort()
            self._download_worker.wait(2000)
        self._stop_worker()
        super().closeEvent(event)

    def _handle_key(self, key: int) -> bool:
        if key == Qt.Key.Key_Right:
            self._step_frame(1)
        elif key == Qt.Key.Key_Left:
            self._step_frame(-1)
        elif key == Qt.Key.Key_Down:
            self._step_burst(1)
        elif key == Qt.Key.Key_Up:
            self._step_burst(-1)
        elif key == Qt.Key.Key_Home:
            self._select_frame(0)
        elif key == Qt.Key.Key_End:
            burst = self._current_burst()
            if burst is None or not burst.frames:
                return False
            self._select_frame(len(burst.frames) - 1)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._go_to_recommended()
        elif Qt.Key.Key_0 <= key <= Qt.Key.Key_5:
            self._set_stars(key - Qt.Key.Key_0)
        elif key == Qt.Key.Key_A:
            self._toggle_decision("keep")
        elif key == Qt.Key.Key_X:
            self._toggle_decision("reject")
        elif key == Qt.Key.Key_R:
            self._reject_burst()
        elif key == Qt.Key.Key_C:
            self._set_checked(self.compare_action, not self._compare)
        elif key == Qt.Key.Key_Z:
            self._set_checked(self.loupe_action, not self._loupe)
        elif key == Qt.Key.Key_F:
            self._set_checked(self.peak_action, not self._peaking)
        elif key == Qt.Key.Key_G:
            self._reject_obvious()
        elif key == Qt.Key.Key_Escape and (self._scanning or self._downloading):
            self._cancel_scan()
        else:
            return False
        return True

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 8)
        root.setSpacing(8)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("Outburst")
        title.setObjectName("title")
        subtitle = QLabel("Offline burst culling for wildlife and nature")
        subtitle.setObjectName("subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        header.addLayout(titles)
        header.addStretch(1)
        self.folder_label = QLabel("No folder open")
        self.folder_label.setObjectName("folderLabel")
        self.folder_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.folder_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        header.addWidget(self.folder_label, 1)
        root.addLayout(header)

        self.burst_list = QListWidget()
        self.burst_list.setObjectName("burstList")
        self.burst_list.setMinimumHeight(168)
        self.burst_list.currentRowChanged.connect(self._on_burst_row)
        self.controls = ControlPanel()
        self.controls.threshold_changed.connect(self._apply_threshold)
        self.controls.weights_changed.connect(self._apply_weights)
        self.controls.subject_toggled.connect(self._on_subject_toggled)
        self.controls.eye_toggled.connect(self._on_eye_toggled)
        self.controls.model_changed.connect(self._on_model_changed)
        self.controls.minimum_changed.connect(self._on_minimum_changed)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        bursts_label = QLabel("Bursts")
        bursts_label.setObjectName("subtitle")
        left_layout.addWidget(bursts_label)
        left_layout.addWidget(self.burst_list, 1)
        left_layout.addWidget(self.controls)

        self.viewer = ImageViewer()
        self.viewer.partner_clicked.connect(self._select_partner)
        self.filmstrip = Filmstrip()
        self.filmstrip.selected.connect(self._select_frame)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(left)
        top.addWidget(self.viewer)
        top.setStretchFactor(0, 0)
        top.setStretchFactor(1, 1)
        top.setSizes([320, 900])
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(self.filmstrip)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Open a folder to start")

    def _build_actions(self) -> None:
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)
        style = self.style()
        open_icon = style.standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)
        refresh_icon = style.standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        save_icon = style.standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)

        file_menu = self.menuBar().addMenu("File")
        view_menu = self.menuBar().addMenu("View")
        help_menu = self.menuBar().addMenu("Help")

        self.open_action = self._action("Open folder…", self._open_folder_dialog, "Ctrl+O", open_icon)
        self.rescan_action = self._action("Rescan", self.rescan, "F5", refresh_icon)
        self.export_action = self._action("Save keepers…", self._export_keepers, "Ctrl+E", save_icon)
        self.report_action = self._action("Export report…", self._export_report, "Ctrl+Shift+E")
        self.xmp_action = self._action("Write XMP sidecars…", self._write_xmp, "Ctrl+Shift+X")
        self.cancel_action = self._action("Cancel scan", self._cancel_scan, "Esc")
        quit_action = self._action("Quit", self.close, "Ctrl+Q")
        self.warnings_action = self._action("Scan notes", self._show_warnings)
        cache_action = self._action("Open cache folder", self._open_cache_folder)
        self.clear_cache_action = self._action("Clear cache…", self._clear_cache)
        self.clear_cache_action.setToolTip(
            "Delete saved measurements and preview images for every folder. Keeper marks stay."
        )
        shortcuts_action = self._action("Keyboard shortcuts", lambda: show_shortcuts(self))
        about_action = self._action("About Outburst", self._about)

        self.open_action.setIconText("Open")
        self.rescan_action.setIconText("Rescan")
        self.rescan_action.setToolTip(
            "Clear cached measurements for this folder and scan it again. "
            "With subject scoring on, Rescan downloads the selected model if it is missing and scores with it. "
            "Keeper marks stay."
        )
        self.export_action.setIconText("Save")
        self.export_action.setToolTip("Save the frames marked as keepers. Originals stay where they are.")
        self.report_action.setIconText("Report")
        self.xmp_action.setIconText("XMP")
        for action in (
            self.open_action,
            self.rescan_action,
            self.export_action,
            self.report_action,
            self.xmp_action,
        ):
            file_menu.addAction(action)
            toolbar.addAction(action)
        file_menu.addSeparator()
        file_menu.addAction(self.cancel_action)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        self.compare_action = self._check_action(
            "Compare with best",
            "Show this frame beside the best other frame in the burst. Keyboard: C",
        )
        self.loupe_action = self._check_action(
            "1:1 loupe",
            "Show the embedded preview at full size and drag to look around. Keyboard: Z",
        )
        self.peak_action = self._check_action(
            "Focus peaking",
            "Paint the strongest edges green. Keyboard: F",
        )
        self.compare_action.toggled.connect(self._on_compare_toggled)
        self.loupe_action.toggled.connect(self._on_loupe_toggled)
        self.peak_action.toggled.connect(self._on_peak_toggled)
        view_menu.addAction(self.compare_action)
        view_menu.addAction(self.loupe_action)
        view_menu.addAction(self.peak_action)
        view_menu.addSeparator()
        view_menu.addAction(self.warnings_action)
        view_menu.addAction(cache_action)
        view_menu.addAction(self.clear_cache_action)
        help_menu.addAction(shortcuts_action)
        help_menu.addAction(about_action)
        toolbar.addAction(self.cancel_action)

    def _action(
        self,
        text: str,
        slot,
        shortcut: str | None = None,
        icon=None,  # noqa: ANN001
    ) -> QAction:
        action = QAction(text, self)
        if icon is not None and not icon.isNull():
            action.setIcon(icon)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False, callback=slot: callback())
        return action

    def _check_action(self, text: str, tooltip: str) -> QAction:
        action = QAction(text, self)
        action.setCheckable(True)
        action.setToolTip(tooltip)
        return action

    def _set_checked(self, action: QAction, checked: bool) -> None:
        action.setChecked(checked)

    def _open_folder_dialog(self) -> None:
        if os.environ.get("OUTBURST_HEADLESS") == "1":
            return
        start = str(self._folder or self._dialog_start)
        chosen = QFileDialog.getExistingDirectory(self, "Open a folder of bursts", start)
        if chosen:
            self.open_path(Path(chosen))

    def _start_worker(self) -> None:
        if self._folder is None:
            return
        if self.controls.prefer_subject() and not self._model_ready():
            self._begin_download(then_scan=True)
            return
        if self._downloading:
            self._download_then_scan = True
            return
        self._scan_id += 1
        scan_id = self._scan_id
        subject_model = self._model_path if self.controls.prefer_subject() else None
        worker = AnalyzeWorker(
            self._folder,
            recursive=self.controls.recursive(),
            threshold=self.controls.threshold(),
            weights=self.controls.weights(),
            cache_dir=self._cache_dir,
            subject_model=subject_model,
            eye_scoring=self.controls.eye_scoring(),
        )
        worker.progress.connect(self._on_progress)
        worker.succeeded.connect(lambda library, ident=scan_id: self._on_success(ident, library))
        worker.failed.connect(lambda message, ident=scan_id: self._on_failed(ident, message))
        worker.cancelled.connect(lambda ident=scan_id: self._on_cancelled(ident))
        self._worker = worker
        self._scanning = True
        self._sync_actions()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.statusBar().showMessage(f"Scanning {self._folder.name}…")
        worker.start()

    def _stop_worker(self) -> bool:
        self._scan_id += 1
        worker = self._worker
        if worker is None:
            self._scanning = False
            return True
        worker.abort()
        if worker.isRunning() and not worker.wait(10_000):
            return False
        self._worker = None
        self._scanning = False
        return True

    def _cancel_scan(self) -> None:
        if self._downloading and self._download_worker is not None:
            self._download_worker.abort()
            self.statusBar().showMessage("Cancelling download…")
            return
        if self._scanning:
            self._stop_worker()
            self.statusBar().showMessage("Scan cancelled")
            self._sync_actions()

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(min(max(done, 0), max(total, 1)))
        self.statusBar().showMessage(message)

    def _on_success(self, scan_id: int, library: object) -> None:
        if scan_id != self._scan_id or not isinstance(library, Library):
            return
        self._scanning = False
        library = with_weights(library, self.controls.weights())
        library = with_threshold(library, self.controls.threshold())
        self.present(library)

    def _on_failed(self, scan_id: int, message: str) -> None:
        if scan_id != self._scan_id:
            return
        self._scanning = False
        self._sync_actions()
        self._notify("Scan failed", message, "warning")

    def _on_cancelled(self, scan_id: int) -> None:
        if scan_id != self._scan_id:
            return
        self._scanning = False
        self._sync_actions()
        self.statusBar().showMessage("Scan cancelled")

    def _apply_threshold(self, value: float) -> None:
        if self._library is None:
            return
        selected = self._selected_path()
        self._clear_automatic_keepers()
        self._library = with_threshold(self._library, value)
        self._mark_keepers_for_new_bursts()
        self._refresh_burst_list(prefer_path=selected)

    def _apply_weights(self) -> None:
        if self._library is None:
            return
        self._library = with_weights(self._library, self.controls.weights())
        self._by_path = {item.frame.path: item for item in self._library.analyses}
        self._refresh_burst_labels()
        self.filmstrip.refresh_flags(self._thumbs())
        self._update_viewer()

    def _on_subject_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._download_then_scan = False
            self._apply_weights()
            return
        if self._model_ready():
            if self._folder is not None:
                self.rescan()
            return
        self._begin_download(then_scan=self._folder is not None)

    def _on_model_changed(self, _model_id: str) -> None:
        if not self.controls.prefer_subject():
            self.controls.set_subject_checked(True)
        if self._folder is None:
            if not self._model_ready():
                self._begin_download(then_scan=False)
            return
        self.rescan()

    def _model_ready(self) -> bool:
        option = option_by_id(self.controls.model_id())
        if option is None:
            self._model_path = None
            return False
        path = model_path(option)
        if model_is_ready(path):
            self._model_path = path
            return True
        self._model_path = None
        return False

    def _begin_download(self, then_scan: bool) -> None:
        if os.environ.get("OUTBURST_HEADLESS") == "1":
            self.controls.set_subject_checked(False)
            self.statusBar().showMessage("Subject model is not in the models folder")
            if then_scan and self._folder is not None:
                self._start_worker()
            return
        option = option_by_id(self.controls.model_id())
        if option is None:
            return
        if self._downloading:
            if self._download_model_id == option.id:
                self._download_then_scan = self._download_then_scan or then_scan
                return
            self._replace_download = True
            self._download_then_scan = then_scan
            if self._download_worker is not None:
                self._download_worker.abort()
            return
        self._download_then_scan = then_scan
        self._download_model_id = option.id
        self._downloading = True
        worker = ModelFetchWorker(option.id, models_dir())
        worker.progress.connect(self._on_download_progress)
        worker.succeeded.connect(self._on_download_succeeded)
        worker.failed.connect(self._on_download_failed)
        worker.cancelled.connect(self._on_download_cancelled)
        self._download_worker = worker
        self.controls.set_model_enabled(False)
        self._sync_actions()
        self.progress.setRange(0, 0)
        self.statusBar().showMessage(f"Downloading {option.label}…")
        worker.start()

    def _on_download_progress(self, done: int, total: int) -> None:
        if total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(min(done, total))
        self.statusBar().showMessage(f"Downloading model… {done / (1024 * 1024):.1f} MB")

    def _on_download_succeeded(self, path: str) -> None:
        then_scan = self._download_then_scan
        self._finish_download_ui()
        self._model_path = Path(path)
        self.controls.refresh_models()
        if then_scan and self.controls.prefer_subject() and self._folder is not None:
            self._start_worker()
            return
        self.statusBar().showMessage(f"Saved {Path(path).name} in the models folder")

    def _on_download_failed(self, message: str) -> None:
        then_scan = self._download_then_scan
        self._finish_download_ui()
        self._notify("Could not download the model", message, "warning")
        if self.controls.prefer_subject():
            self.controls.set_subject_checked(False)
        if then_scan and self._folder is not None:
            self._start_worker()

    def _on_download_cancelled(self) -> None:
        replace = self._replace_download
        then_scan = self._download_then_scan
        self._replace_download = False
        self._finish_download_ui()
        if replace:
            self._begin_download(then_scan)
            return
        self.statusBar().showMessage("Model download cancelled")
        if self.controls.prefer_subject():
            self.controls.set_subject_checked(False)
        if then_scan and self._folder is not None:
            self._start_worker()

    def _finish_download_ui(self) -> None:
        worker = self._download_worker
        self._downloading = False
        self._download_then_scan = False
        self._download_worker = None
        if worker is not None and worker.isRunning():
            worker.wait(2000)
        self.controls.set_model_enabled(True)
        self._sync_actions()

    def _on_minimum_changed(self, _value: int) -> None:
        self._refresh_burst_list(prefer_path=self._selected_path())
        if self._library is not None:
            self.statusBar().showMessage(self._status_summary())

    def _visible_bursts(self) -> tuple[Burst, ...]:
        if self._library is None:
            return ()
        return visible_bursts(self._library.bursts, minimum_frames=self.controls.minimum_frames())

    def _status_summary(self) -> str:
        library = self._library
        if library is None:
            return ""
        shown = len(self._visible_bursts())
        hidden = len(library.bursts) - shown
        noun = "burst" if shown == 1 else "bursts"
        message = f"{len(library.analyses)} files · {shown} {noun}"
        if hidden:
            hidden_noun = "burst" if hidden == 1 else "bursts"
            message += (
                f" · {hidden} {hidden_noun} under {self.controls.minimum_frames()} frames hidden"
            )
        if library.warnings:
            message += f" · {len(library.warnings)} notes"
        modes: list[str] = []
        if self._compare:
            modes.append("compare")
        if self._loupe:
            modes.append("1:1 loupe")
        if self._peaking:
            modes.append("focus peaking")
        if library.eye_active:
            message += " · eye scoring"
        if modes:
            message += " · " + " · ".join(modes)
        return message

    def _on_burst_row(self, row: int) -> None:
        if row < 0 or row >= len(self._visible_bursts()):
            return
        self._show_burst(row, prefer_path=None)

    def _refresh_burst_list(self, prefer_path: Path | None) -> None:
        library = self._library
        self.burst_list.blockSignals(True)
        self.burst_list.clear()
        if library is None:
            self.burst_list.blockSignals(False)
            self.viewer.set_placeholder(_WELCOME)
            return
        for burst in self._visible_bursts():
            item = QListWidgetItem(burst_list_text(burst, self._by_path, self._decisions))
            item.setData(Qt.ItemDataRole.UserRole, burst.id)
            item.setSizeHint(QSize(240, 58))
            self.burst_list.addItem(item)
        row = self._row_for_path(prefer_path)
        if self.burst_list.count():
            self.burst_list.setCurrentRow(row)
        self.burst_list.blockSignals(False)
        if self.burst_list.count():
            self._show_burst(row, prefer_path=prefer_path)
            return
        self._burst_index = 0
        self._frame_index = 0
        self.filmstrip.set_cards([])
        if library.bursts:
            self.viewer.set_placeholder(_short_bursts_hidden(self.controls.minimum_frames()))
        else:
            self.viewer.set_placeholder(_WELCOME)

    def _refresh_burst_labels(self) -> None:
        if self._library is None:
            return
        for row, burst in enumerate(self._visible_bursts()):
            item = self.burst_list.item(row)
            if item is not None:
                item.setText(burst_list_text(burst, self._by_path, self._decisions))

    def _show_burst(self, index: int, prefer_path: Path | None) -> None:
        bursts = self._visible_bursts()
        if not (0 <= index < len(bursts)):
            return
        self._burst_index = index
        burst = bursts[index]
        frame_index = None
        if prefer_path is not None:
            frame_index = next((i for i, frame in enumerate(burst.frames) if frame.path == prefer_path), None)
        if frame_index is None and burst.frames:
            frame_index = recommended_offset(burst.frames, self._by_path)
        self._frame_index = frame_index or 0
        self.filmstrip.set_cards(self._thumbs())
        self._update_viewer()

    def _select_frame(self, index: int) -> None:
        burst = self._current_burst()
        if burst is None or not burst.frames:
            return
        self._frame_index = max(0, min(index, len(burst.frames) - 1))
        self.filmstrip.mark_selected(self._frame_index)
        self._update_viewer()

    def _step_frame(self, delta: int) -> None:
        self._select_frame(self._frame_index + delta)

    def _step_burst(self, delta: int) -> None:
        count = len(self._visible_bursts())
        if count == 0:
            return
        row = max(0, min(self._burst_index + delta, count - 1))
        self.burst_list.setCurrentRow(row)

    def _go_to_recommended(self) -> None:
        burst = self._current_burst()
        if burst is None or not burst.frames:
            return
        self._select_frame(recommended_offset(burst.frames, self._by_path))

    def _toggle_decision(self, kind: str) -> None:
        analysis = self._current_analysis()
        if analysis is None or self._scanning:
            return
        path = analysis.frame.path
        updated = None if self._decisions.get(path) == kind else kind
        self._apply_decision(path, updated, origin="user")
        self._refresh_burst_labels()
        self.filmstrip.refresh_flags(self._thumbs())
        self._update_viewer()

    def _reject_burst(self) -> None:
        burst = self._current_burst()
        if burst is None or self._scanning:
            return
        clear = all(self._decisions.get(frame.path) == "reject" for frame in burst.frames)
        for frame in burst.frames:
            self._apply_decision(frame.path, None if clear else "reject", origin="user")
        self._refresh_burst_labels()
        self.filmstrip.refresh_flags(self._thumbs())
        self._update_viewer()

    def _on_compare_toggled(self, checked: bool) -> None:
        self._compare = checked
        self._update_viewer()
        self._show_mode_status()

    def _on_loupe_toggled(self, checked: bool) -> None:
        self._loupe = checked
        self.viewer.set_view_mode(zoom=self._loupe, peak=self._peaking)
        self._show_mode_status()

    def _on_peak_toggled(self, checked: bool) -> None:
        self._peaking = checked
        self.viewer.set_view_mode(zoom=self._loupe, peak=self._peaking)
        self._show_mode_status()

    def _show_mode_status(self) -> None:
        if self._library is not None:
            self.statusBar().showMessage(self._status_summary())

    def _select_partner(self) -> None:
        burst = self._current_burst()
        if burst is None:
            return
        partner = compare_partner_offset(burst.frames, self._by_path, self._frame_index)
        if partner is None:
            return
        self._select_frame(partner)

    def _reject_obvious(self) -> None:
        """Reject soft or badly clipped frames, leaving the best frame and your keepers."""

        burst = self._current_burst()
        if burst is None or self._scanning or self._library is None:
            return
        paths = {frame.path for frame in burst.frames}
        remembered = [path for path in self._obvious_rejects if path in paths]
        if (
            self._obvious_burst_id == burst.id
            and remembered
            and all(self._decisions.get(path) == "reject" for path in remembered)
        ):
            for path in remembered:
                self._apply_decision(path, None)
            self._obvious_rejects = []
            self._obvious_burst_id = None
            self._refresh_burst_labels()
            self.filmstrip.refresh_flags(self._thumbs())
            self._update_viewer()
            self.statusBar().showMessage("Restored the frames rejected as obvious misses")
            return
        misses = obvious_misses(burst.frames, self._by_path, self._decisions)
        if not misses:
            self.statusBar().showMessage("No obvious misses in this burst")
            return
        for path in misses:
            self._apply_decision(path, "reject")
        self._obvious_rejects = list(misses)
        self._obvious_burst_id = burst.id
        noun = "frame" if len(misses) == 1 else "frames"
        self._refresh_burst_labels()
        self.filmstrip.refresh_flags(self._thumbs())
        self._update_viewer()
        self.statusBar().showMessage(
            f"Rejected {len(misses)} obvious {noun}. Press G again to restore them."
        )

    def _update_viewer(self) -> None:
        analysis = self._current_analysis()
        burst = self._current_burst()
        if analysis is None or burst is None or self._library is None:
            self.viewer.set_placeholder(_WELCOME)
            return
        frame = analysis.frame
        recommended_index = recommended_offset(burst.frames, self._by_path) if burst.frames else 0
        recommended = self._frame_index == recommended_index
        decision = self._decisions.get(frame.path)
        show_subject = self._library.subject_active and self._library.weights.prefer_subject
        box = None
        if show_subject and analysis.metrics is not None:
            box = analysis.metrics.subject_box
        html = frame_info_html(
            filename=frame.path.name,
            frame_number=self._frame_index + 1,
            frame_count=burst.count,
            burst_id=burst.id,
            captured_at=frame.captured_at,
            camera=frame.camera,
            analysis=analysis,
            weights=self._library.weights,
            recommended=recommended,
            decision=decision,
            subject_active=show_subject,
            stars=self._stars.get(frame.path),
            eye_active=self._library.eye_active,
        )
        keeper = decision == "keep"
        rejected = decision == "reject"
        if rejected:
            border = "#8a3a3a"
        elif keeper:
            border = "#3dbe86"
        elif recommended:
            border = "#c9a227"
        else:
            border = "#2e2e2e"
        self.viewer.set_frame(
            analysis.preview_path,
            box,
            html,
            border,
            keeper=keeper,
            rejected=rejected,
            stars=self._stars.get(frame.path),
            circle=self._eye_circle(analysis),
        )
        self.viewer.set_view_mode(zoom=self._loupe, peak=self._peaking)
        self.viewer.set_compare(self._compare)
        if not self._compare:
            return
        partner_index = compare_partner_offset(burst.frames, self._by_path, self._frame_index)
        if partner_index is None:
            self.viewer.set_partner_message("Compare", "This burst has one frame.")
            return
        partner_frame = burst.frames[partner_index]
        partner = self._by_path.get(partner_frame.path)
        partner_decision = self._decisions.get(partner_frame.path)
        partner_box = None
        if (
            show_subject
            and partner is not None
            and partner.metrics is not None
        ):
            partner_box = partner.metrics.subject_box
        role = "Best in the burst" if partner_index == recommended_index else "Next sharpest"
        score = score_caption(
            None if partner is None else partner.composite,
            None if partner is None else partner.metrics,
            best_composite=None,
        )
        self.viewer.set_partner(
            None if partner is None else partner.preview_path,
            partner_box,
            f"{role}  ·  {score}  ·  {partner_frame.path.name}",
            circle=None if partner is None else self._eye_circle(partner),
            keeper=partner_decision == "keep",
            rejected=partner_decision == "reject",
        )

    def _thumbs(self) -> list[Thumb]:
        burst = self._current_burst()
        if burst is None or self._library is None:
            return []
        best = recommended_offset(burst.frames, self._by_path) if burst.frames else 0
        best_analysis = self._by_path.get(burst.frames[best].path) if burst.frames else None
        best_composite = None if best_analysis is None else best_analysis.composite
        cards: list[Thumb] = []
        for index, frame in enumerate(burst.frames):
            analysis = self._by_path.get(frame.path)
            decision = self._decisions.get(frame.path)
            name = frame.path.name
            title = name if len(name) <= 22 else f"{name[:9]}…{name[-10:]}"
            cards.append(
                Thumb(
                    title=title,
                    score_text=score_caption(
                        None if analysis is None else analysis.composite,
                        None if analysis is None else analysis.metrics,
                        best_composite=None if index == best else best_composite,
                        stars=self._stars.get(frame.path),
                    ),
                    badge=thumb_badge(index == best, decision),
                    thumb_path=None if analysis is None else analysis.thumb_path,
                    selected=index == self._frame_index,
                    best=index == best,
                    decision=decision,
                    tooltip=name,
                )
            )
        return cards

    def _export_keepers(self) -> None:
        if self._library is None:
            return
        request = ExportKeepersDialog.ask(self, self._folder)
        if request is None:
            return
        minimum = self.controls.minimum_frames()
        choices = choose_keepers(
            self._library.bursts,
            self._by_path,
            self._decisions,
            minimum_frames=minimum,
        )
        if not choices:
            self._notify(
                "No keepers to save",
                f"Mark a keeper in a burst of at least {minimum} frames. Shorter bursts are hidden and are not saved.",
                "info",
            )
            return
        try:
            results = export_keepers([choice.path for choice in choices], request.folder, request.mode)
        except OSError as exc:
            self._notify("Save failed", str(exc), "warning")
            return
        notes = [result.note for result in results if result.note]
        message = f"Saved {len(results)} keeper(s) to {request.folder}"
        if notes:
            message += " " + notes[0]
        self.statusBar().showMessage(message)

    def _export_report(self) -> None:
        if self._library is None or os.environ.get("OUTBURST_HEADLESS") == "1":
            return
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Export a report",
            str(self._folder or self._dialog_start),
        )
        if not chosen:
            return
        try:
            json_path, csv_path = write_reports(
                self._library,
                self._decisions,
                Path(chosen),
                stars=self._stars,
                minimum_frames=self.controls.minimum_frames(),
            )
        except OSError as exc:
            self._notify("Report failed", str(exc), "warning")
            return
        self.statusBar().showMessage(f"Wrote {json_path.name} and {csv_path.name}")

    def _write_xmp(self) -> None:
        if self._library is None:
            return
        count = len(self._library.analyses)
        if not ask(
            self,
            "Write XMP sidecars",
            f"Write sidecars next to {count} files?\n\n"
            "Each sidecar stores the star rating and the pick or reject flag for Lightroom, "
            "Capture One, Darktable, and digiKam. Original RAW and JPEG files stay unchanged.",
        ):
            return
        written, warnings = write_library_sidecars(
            self._library,
            self._decisions,
            stars=self._stars,
            minimum_frames=self.controls.minimum_frames(),
        )
        message = f"Wrote {len(written)} sidecar(s)."
        if warnings:
            message += " " + warnings[0]
        self.statusBar().showMessage(message)
        if warnings:
            self._warnings = tuple(warnings)
            self._sync_actions()

    def _show_warnings(self) -> None:
        show_warnings(self, self._warnings)

    def _open_cache_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._cache_dir)))

    def _clear_cache(self) -> None:
        if self._scanning or self._downloading:
            self._notify("Scan in progress", "Cancel the scan before clearing the cache.", "warning")
            return
        if not ask(
            self,
            "Clear cache",
            "Remove saved measurements and preview images for every folder?\n\n"
            "Keeper and reject marks stay. If a folder is open, Outburst measures it again.",
        ):
            return
        try:
            with Cache(self._cache_dir) as cache:
                removed = cache.clear()
        except OSError as exc:
            self._notify("Could not clear the cache", str(exc), "warning")
            return
        self.statusBar().showMessage(f"Cleared cached measurements for {removed} files")
        if self._folder is not None:
            self.rescan()

    def _forget_folder_cache(self, folder: Path) -> int:
        try:
            with Cache(self._cache_dir) as cache:
                return cache.forget_folder(folder)
        except OSError as exc:
            log.warning("Could not clear cached measurements for %s: %s", folder, exc)
            self._notify(
                "Could not clear the cache",
                f"Outburst will rescan, but some saved measurements may be reused. {exc}",
                "warning",
            )
            return 0

    def _mark_recommended_keepers(self) -> None:
        """Mark the recommended frame when a burst does not already have a keeper.

        When two or more frames have a rated eye, the sharpest eye is the
        frame Outburst marks. Earlier automatic keepers in that burst are
        cleared. A reject stays rejected, and a keeper the photographer set
        keeps its origin.
        """

        if self._library is None:
            return
        for burst in self._library.bursts:
            if not burst.frames:
                continue
            offset = recommended_offset(burst.frames, self._by_path)
            frame = burst.frames[offset]
            if eye_chooses_best(burst.frames, self._by_path):
                for other in burst.frames:
                    if other.path == frame.path:
                        continue
                    if (
                        self._decisions.get(other.path) == "keep"
                        and self._origins.get(other.path) == "auto"
                    ):
                        self._apply_decision(other.path, None)
                if self._decisions.get(frame.path) == "reject":
                    continue
                if self._decisions.get(frame.path) == "keep":
                    continue
                self._apply_decision(frame.path, "keep", origin="auto")
                continue
            if any(self._decisions.get(item.path) == "keep" for item in burst.frames):
                continue
            if self._decisions.get(frame.path) == "reject":
                continue
            self._apply_decision(frame.path, "keep", origin="auto")

    def _on_eye_toggled(self, enabled: bool) -> None:
        del enabled
        self._clear_automatic_keepers()
        if self._folder is not None:
            self.rescan()

    def _eye_circle(self, analysis: FrameAnalysis) -> tuple[int, int, int] | None:
        if self._library is None or not self._library.eye_active:
            return None
        metrics = analysis.metrics
        if metrics is None or metrics.eye_center is None or metrics.eye_radius is None:
            return None
        x, y = metrics.eye_center
        return (int(x), int(y), int(metrics.eye_radius))

    def _clear_automatic_keepers(self) -> None:
        """Drop keepers Outburst chose. Marks the photographer set stay."""

        automatic = [
            path
            for path, origin in self._origins.items()
            if origin == "auto" and self._decisions.get(path) == "keep"
        ]
        for path in automatic:
            self._apply_decision(path, None)

    def _mark_keepers_for_new_bursts(self) -> None:
        """Mark each burst's best frame after the burst gap changes.

        A frame the photographer rejected stays rejected. A frame they already
        marked as a keeper keeps that mark. Every other best frame is marked
        automatically.
        """

        if self._library is None:
            return
        for burst in self._library.bursts:
            if not burst.frames:
                continue
            frame = burst.frames[recommended_offset(burst.frames, self._by_path)]
            current = self._decisions.get(frame.path)
            if current == "reject":
                continue
            if current == "keep" and self._origins.get(frame.path) != "auto":
                continue
            self._apply_decision(frame.path, "keep", origin="auto")

    def _about(self) -> None:
        if os.environ.get("OUTBURST_HEADLESS") == "1":
            return
        QMessageBox.about(
            self,
            "About Outburst",
            "<h2>Outburst</h2>"
            f"<p>Version {__version__}</p>"
            "<p>Outburst is a free, offline burst-culling app for wildlife and nature photographers. "
            "Your photos stay on this computer. Outburst never modifies the original files. "
            "A subject model is downloaded into the models folder only when you turn that "
            "feature on and the weights are not already there.</p>",
        )

    def _current_burst(self) -> Burst | None:
        bursts = self._visible_bursts()
        if not bursts or not (0 <= self._burst_index < len(bursts)):
            return None
        return bursts[self._burst_index]

    def _current_analysis(self) -> FrameAnalysis | None:
        burst = self._current_burst()
        if burst is None or not burst.frames:
            return None
        index = max(0, min(self._frame_index, len(burst.frames) - 1))
        return self._by_path.get(burst.frames[index].path)

    def _selected_path(self) -> Path | None:
        analysis = self._current_analysis()
        if analysis is None:
            return None
        return analysis.frame.path

    def _row_for_path(self, path: Path | None) -> int:
        if self._library is None or path is None:
            return 0
        for index, burst in enumerate(self._visible_bursts()):
            if any(frame.path == path for frame in burst.frames):
                return index
        return 0

    def _load_decisions(self) -> None:
        if self._library is None:
            self._decisions = {}
            self._origins = {}
            self._stars = {}
            return
        paths = [item.frame.path for item in self._library.analyses]
        try:
            with Cache(self._cache_dir) as cache:
                self._decisions = cache.decisions_for(paths)
                self._origins = cache.decision_origins(paths)
                self._stars = cache.stars_for(paths)
        except OSError as exc:
            log.warning("Could not read saved decisions: %s", exc)
            self._decisions = {}
            self._origins = {}
            self._stars = {}

    def _apply_decision(self, path: Path, decision: str | None, *, origin: str = "user") -> None:
        if decision is None:
            self._decisions.pop(path, None)
            self._origins.pop(path, None)
            stored_origin = "user"
        else:
            stored_origin = origin if decision == "keep" else "user"
            self._decisions[path] = decision
            self._origins[path] = stored_origin
        self._save_decision(path, decision, origin=stored_origin)

    def _set_stars(self, stars: int) -> None:
        analysis = self._current_analysis()
        if analysis is None or self._scanning:
            return
        path = analysis.frame.path
        if stars == 0 or self._stars.get(path) == stars:
            updated = None
            self._stars.pop(path, None)
        else:
            updated = stars
            self._stars[path] = stars
        self._save_stars(path, updated)
        self._refresh_burst_labels()
        self.filmstrip.refresh_flags(self._thumbs())
        self._update_viewer()

    def _save_stars(self, path: Path, stars: int | None) -> None:
        try:
            with Cache(self._cache_dir) as cache:
                cache.set_stars(path, stars)
        except OSError as exc:
            self._notify("Could not save that rating", str(exc), "warning")

    def _save_decision(self, path: Path, decision: str | None, *, origin: str = "user") -> None:
        try:
            with Cache(self._cache_dir) as cache:
                cache.set_decision(path, decision, origin=origin)
        except OSError as exc:
            self._notify("Could not save that decision", str(exc), "warning")

    def _set_folder_label(self) -> None:
        if self._folder is None:
            self.folder_label.setText("No folder open")
            self.folder_label.setToolTip("")
            return
        self.folder_label.setText(str(self._folder))
        self.folder_label.setToolTip(str(self._folder))

    def _sync_actions(self) -> None:
        busy = self._scanning or self._downloading
        ready = self._library is not None and not self._scanning
        self.export_action.setEnabled(ready)
        self.report_action.setEnabled(ready)
        self.xmp_action.setEnabled(ready)
        self.rescan_action.setEnabled(self._folder is not None and not busy)
        self.cancel_action.setEnabled(busy)
        self.cancel_action.setVisible(busy)
        self.cancel_action.setText("Cancel download" if self._downloading else "Cancel scan")
        self.warnings_action.setEnabled(bool(self._warnings))
        self.progress.setVisible(busy)

    def _notify(self, title: str, text: str, level: str) -> None:
        notify(self, title, text, level)

    def _load_settings(self) -> None:
        settings = QSettings("Outburst", "Outburst")
        self.controls.set_threshold(_as_float(settings.value("threshold"), 1.0))
        self.controls.set_sharpness_percent(_as_int(settings.value("sharpness"), 75))
        self.controls.set_center_percent(_as_int(settings.value("center_mix"), 65))
        self.controls.set_recursive(_as_bool(settings.value("recursive"), False))
        self.controls.set_minimum_frames(_saved_minimum_frames(settings))
        self.controls.set_subject_checked(_as_bool(settings.value("prefer_subject"), False))
        self.controls.set_eye_scoring(_as_bool(settings.value("eye_scoring"), False))
        folder = settings.value("last_folder", "")
        if folder:
            self._dialog_start = Path(str(folder))
        self.controls.set_model_id(_saved_model_id(settings))
        self._model_ready()

    def _save_settings(self) -> None:
        settings = QSettings("Outburst", "Outburst")
        settings.setValue("threshold", self.controls.threshold())
        settings.setValue("sharpness", self.controls.sharpness_percent())
        settings.setValue("center_mix", self.controls.center_percent())
        settings.setValue("recursive", self.controls.recursive())
        settings.setValue("minimum_frames", self.controls.minimum_frames())
        settings.setValue("prefer_subject", self.controls.prefer_subject())
        settings.setValue("eye_scoring", self.controls.eye_scoring())
        settings.setValue("subject_model_id", self.controls.model_id())
        if self._folder is not None:
            settings.setValue("last_folder", str(self._folder))
        if self._model_path is not None:
            settings.setValue("subject_model", str(self._model_path))


def _typing_target(widget: QWidget | None) -> bool:
    return isinstance(widget, (QLineEdit, QAbstractSpinBox, QAbstractSlider, QComboBox))


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _saved_minimum_frames(settings: QSettings) -> int:
    stored = settings.value("minimum_frames")
    if stored not in (None, ""):
        return max(1, _as_int(stored, 2))
    if _as_bool(settings.value("show_single_bursts"), False):
        return 1
    return 2


def _saved_model_id(settings: QSettings) -> str:
    model_id = str(settings.value("subject_model_id", "") or "")
    if option_by_id(model_id) is not None:
        return model_id
    legacy = str(settings.value("subject_model", "") or "")
    name = Path(legacy).name if legacy else ""
    for option in MODEL_OPTIONS:
        if option.filename == name:
            return option.id
    return DEFAULT_MODEL_ID


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes"}
