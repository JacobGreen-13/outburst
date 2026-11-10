"""The Outburst review window.

``MainWindow.open_path`` and ``rescan`` scan a folder and clear the burst list, preview, and filmstrip until that scan finishes. ``present`` shows a finished library. ``_select_frame``, ``_step_frame``, and ``_show_burst`` move through the filmstrip order. ``_toggle_decision``, ``_set_stars``, and ``_reject_obvious`` record picks, stars, and rejects. ``_export_keepers``, ``_export_report``, and ``_write_xmp`` write copies, a report, and sidecars. ``_load_settings`` and ``_save_settings`` remember the controls, including filmstrip order. Keyboard handling ignores a focused slider or text box.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from PySide6.QtCore import QEvent, QSettings, QSize, Qt, QThread, QUrl
from PySide6.QtGui import QAction, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
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
    detected_eye_count,
    filmstrip_order,
    frame_info_parts,
    obvious_misses,
    score_caption,
    thumb_badge,
    visible_bursts,
    visible_frame_indexes,
)
from outburst.gui.burst_view import Filmstrip, Thumb
from outburst.gui.controls import ControlPanel
from outburst.gui.dialogs import ExportKeepersDialog, ask, notify, show_shortcuts, show_warnings
from outburst.gui.image_viewer import ImageViewer, next_zoom
from outburst.gui.styles import build_app_icon
from outburst.gui.worker import (
    AnalyzeWorker,
    ExportWorker,
    ModelFetchWorker,
    RuntimeInstallWorker,
    SidecarWorker,
)
from outburst.models import Burst, FrameAnalysis, Library
from outburst.output import choose_keepers, frames_for_sidecars, write_reports
from outburst.pipeline import with_threshold, with_weights
from outburst.scoring import compare_partner_offset, eye_chooses_best, recommended_offset
from outburst.subject import (
    BIRD_EYE_OPTION,
    DEFAULT_MODEL_ID,
    MODEL_OPTIONS,
    model_is_ready,
    model_path,
    models_dir,
    option_by_id,
    subject_runtime_ready,
)

log = logging.getLogger(__name__)

_WELCOME = (
    "Open a folder of RAW bursts to start culling.\n\n"
    "Outburst groups frames shot close together, scores sharpness and exposure "
    "from embedded previews, and marks the recommended frame as a keeper.\n\n"
    "Use Keep under the preview to choose what Save keepers writes. "
    "Original files stay untouched."
)

_SCANNING_PREVIEW = (
    "Scanning and scoring this folder.\n\n"
    "The preview and filmstrip stay empty until the scan finishes."
)


def _short_bursts_hidden(minimum: int, *, eyes: int = 0) -> str:
    """
    Explain that every burst is shorter than the minimum and therefore hidden.

    Inputs: minimum, the Minimum frames setting; eyes, how many frames have a detected bird eye.
    Output: placeholder text that names the minimum and tells the photographer to lower it. A nonzero eye count is included.
    Uses: the preview when a scanned folder has no visible burst.
    Limitations: the text describes the minimum. It does not change the control. An eye too small to rate is still counted.
    """
    text = (
        f"Every burst in this folder has fewer than {minimum} frames.\n\n"
        "Lower Minimum frames to review them."
    )
    if eyes == 1:
        text += "\n\nBird eye detection found an eye on 1 frame."
    elif eyes > 1:
        text += f"\n\nBird eye detection found an eye on {eyes} frames."
    return text


class MainWindow(QMainWindow):
    def __init__(self, cache_dir: Path | None = None) -> None:
        """
        Build the review window, restore settings, and listen for keyboard shortcuts.

        Inputs: cache_dir, where measurements and marks are stored, or None for the default cache.
        Output: no return value. The window is sized, empty of a library, and ready to open a folder.
        Uses: the review window.
        Limitations: no folder is scanned until open_path runs. The application event filter is installed so review keys can be handled before the focused widget.
        """
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
        self._compare_index: int | None = None
        self._loupe = False
        self._zoom_scale: float | None = None
        self._peaking = False
        self._obvious_burst_id: str | None = None
        self._obvious_rejects: list[Path] = []
        self._folder: Path | None = None
        self._dialog_start = Path.home()
        self._model_path: Path | None = None
        self._eye_model_path: Path | None = None
        self._download_worker: ModelFetchWorker | None = None
        self._downloading = False
        self._download_then_scan = False
        self._download_model_id: str | None = None
        self._download_role = "subject"
        self._replace_download = False
        self._burst_index = 0
        self._frame_index = 0
        self._filmstrip_sort = "time"
        self._warnings: tuple[str, ...] = ()
        self._worker: AnalyzeWorker | None = None
        self._scan_id = 0
        self._scanning = False
        self._review_suspended = False
        self._save_worker: QThread | None = None
        self._save_id = 0
        self._saving = False
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
        """
        The library currently shown, if a scan has finished.

        Inputs: none.
        Output: the Library, or None before the first successful scan in this window.
        Uses: tests and the review window state.
        Limitations: this is the in-memory library. It is not a fresh read from disk.
        """
        return self._library

    @property
    def is_scanning(self) -> bool:
        """
        Whether a folder scan is in progress.

        Inputs: none.
        Output: True while this window's scan flag is set.
        Uses: tests and actions that wait for a scan.
        Limitations: a model download and a keeper save are separate flags and are not reported here.
        """
        return self._scanning

    @property
    def current_burst_index(self) -> int:
        """
        The row of the burst on screen.

        Inputs: none.
        Output: the index into the bursts that meet Minimum frames.
        Uses: the burst list selection.
        Limitations: the index can stay 0 when the visible list is empty. Bursts under Minimum frames are not counted.
        """
        return self._burst_index

    @property
    def current_frame_index(self) -> int:
        """
        The frame index inside the current burst.

        Inputs: none.
        Output: the index in shooting order.
        Uses: the preview and the filmstrip selection.
        Limitations: arrow keys follow the on-screen order, but this value stays the frame's shooting index.
        """
        return self._frame_index

    def present(self, library: Library) -> None:
        """
        Show a finished library. Tests and the scan worker both use this.

        Inputs: library, the scanned folder, bursts, analyses, and warnings.
        Output: no return value. The review surface is shown again. Decisions are loaded, recommended keepers are marked, and the burst list, preview, and status summary update.
        Uses: the review window after a scan.
        Limitations: a library with warnings and no analyses shows the first warning as a notice. Original files are not modified.
        """

        self._resume_review()
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
        """
        Open a folder and start a scan.

        Inputs: folder, the directory of bursts. The path is expanded and resolved.
        Output: no return value. A directory is stored and a scan starts. A missing directory, or a previous scan that will not stop, shows a warning.
        Uses: Open folder and a folder passed on the command line.
        Limitations: the path must already be a directory. The folder is left unchanged when the previous scan does not stop within the wait.
        """
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
        """
        Drop cached measurements for the open folder and scan it again.

        Inputs: none.
        Output: no return value. With no folder, nothing happens. Otherwise the preview and filmstrip are cleared, the folder cache is forgotten, and a scan starts, after a missing subject model or bird-eye model is downloaded when that option is on.
        Uses: the Rescan action.
        Limitations: keeper, reject, and star marks stay in the cache. A previous scan that will not stop shows a warning and does not clear the review. The cleared frames stay hidden if the new scan is cancelled. A missing Ultralytics install is installed before the scan when subject or bird eye detection is on.
        """
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
        self._suspend_review()
        self._forget_folder_cache(folder)
        if self.controls.prefer_subject() and not self._model_ready():
            self._begin_download(then_scan=True)
            return
        self.open_path(folder)

    def eventFilter(self, watched, event) -> bool:  # type: ignore[no-untyped-def]
        """
        Handle review shortcuts, and open the folder when its path is clicked.

        Inputs: watched, the object that received the event; event, the Qt event.
        Output: True when a review shortcut or a folder-path click was handled. Otherwise the base filter's result.
        Uses: keyboard review and the toolbar folder path.
        Limitations: keys are ignored while a spin box, slider, line edit, or combo box is focused. Only a KeyPress whose target is the focus widget is considered. A click opens the folder only when one is open.
        """
        if (
            watched is self.folder_label
            and event.type() == QEvent.Type.MouseButtonRelease
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._open_current_folder()
            return True
        if event.type() == QEvent.Type.KeyPress and watched is QApplication.focusWidget():
            if not _typing_target(watched) and self._handle_key(event.key()):
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Handle a key delivered to the window itself.

        Inputs: event, the key press.
        Output: no return value. A handled shortcut is accepted. Any other key goes to the base handler.
        Uses: keyboard review.
        Limitations: keys are ignored while a spin box, slider, line edit, or combo box is focused.
        """
        if not _typing_target(QApplication.focusWidget()) and self._handle_key(event.key()):
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Save settings and stop background work before the window closes.

        Inputs: event, the close event.
        Output: no return value. The event filter is removed, settings are written, a download is aborted, a save is given three seconds, and a scan is stopped. The base close then runs.
        Uses: quitting the review window.
        Limitations: the download wait is two seconds and the save wait is three seconds. The save id changes first, so a late save result is ignored.
        """
        application = QApplication.instance()
        if application is not None:
            application.removeEventFilter(self)
        self._save_settings()
        if self._download_worker is not None and self._download_worker.isRunning():
            self._download_worker.abort()
            self._download_worker.wait(2000)
        self._save_id += 1
        if self._save_worker is not None and self._save_worker.isRunning():
            self._save_worker.wait(3000)
        self._stop_worker()
        super().closeEvent(event)

    def _handle_key(self, key: int) -> bool:
        """
        Run the review shortcut for one key.

        Inputs: key, the Qt key code.
        Output: True when the key is a review shortcut, and False when it is not.
        Uses: frame, burst, rating, compare, zoom, peaking, and cancel shortcuts.
        Limitations: the caller ignores keys while a spin box is focused. Arrow keys follow the on-screen filmstrip order. Home and End use that same order. Escape cancels only while a scan or a download is active. While the review is cleared for a scan, only Escape is handled.
        """
        if self._review_suspended:
            if key == Qt.Key.Key_Escape and (self._scanning or self._downloading):
                self._cancel_scan()
                return True
            return False
        if key == Qt.Key.Key_Right:
            self._step_frame(1)
        elif key == Qt.Key.Key_Left:
            self._step_frame(-1)
        elif key == Qt.Key.Key_Down:
            self._step_burst(1)
        elif key == Qt.Key.Key_Up:
            self._step_burst(-1)
        elif key == Qt.Key.Key_Home:
            order = self._filmstrip_indexes()
            if not order:
                return False
            self._select_frame(order[0])
        elif key == Qt.Key.Key_End:
            order = self._filmstrip_indexes()
            if not order:
                return False
            self._select_frame(order[-1])
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
        elif key == Qt.Key.Key_BracketLeft:
            self._step_compare(-1)
        elif key == Qt.Key.Key_BracketRight:
            self._step_compare(1)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self._step_zoom(1)
        elif key == Qt.Key.Key_Minus:
            self._step_zoom(-1)
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
        """
        Lay out the burst list, controls, preview, and filmstrip.

        Inputs: none.
        Output: no return value. The central widget, splitters, sort buttons, progress bar, and status bar are created.
        Uses: the review window layout.
        Limitations: Time starts checked, so the filmstrip mode begins as shooting order. The progress bar stays hidden until work starts. The initial status says to open a folder.
        """
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        self.burst_list = QListWidget()
        self.burst_list.setObjectName("burstList")
        self.burst_list.setMinimumHeight(120)
        self.burst_list.currentRowChanged.connect(self._on_burst_row)
        self.controls = ControlPanel()
        self.controls.threshold_changed.connect(self._apply_threshold)
        self.controls.weights_changed.connect(self._apply_weights)
        self.controls.subject_toggled.connect(self._on_subject_toggled)
        self.controls.eye_toggled.connect(self._on_eye_toggled)
        self.controls.subject_box_toggled.connect(self._on_subject_box_toggled)
        self.controls.eye_circle_toggled.connect(self._on_eye_circle_toggled)
        self.controls.model_changed.connect(self._on_model_changed)
        self.controls.minimum_changed.connect(self._on_minimum_changed)
        self.controls.minimum_score_changed.connect(self._on_minimum_score_changed)
        bursts = QGroupBox("Bursts")
        bursts.setObjectName("sectionBox")
        bursts_layout = QVBoxLayout(bursts)
        bursts_layout.setContentsMargins(8, 8, 8, 8)
        bursts_layout.addWidget(self.burst_list, 1)
        left = QWidget()
        left.setObjectName("sidePanel")
        left.setMinimumWidth(320)
        left.setMaximumWidth(420)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(bursts, 1)
        left_layout.addWidget(self.controls)

        self.viewer = ImageViewer()
        self.viewer.partner_clicked.connect(self._select_partner)
        self.viewer.zoom_changed.connect(self._on_zoom_changed)
        self.filmstrip = Filmstrip()
        self.filmstrip.selected.connect(self._select_frame)
        self.filmstrip.compare_chosen.connect(self._set_compare_frame)
        self.sort_time = QPushButton("Time")
        self.sort_score = QPushButton("Score")
        self._sort_group = QButtonGroup(self)
        self._sort_group.setExclusive(True)
        for button, mode, tip in (
            (self.sort_time, "time", "Show the burst in the order the frames were shot."),
            (
                self.sort_score,
                "score",
                "Show the highest score first. Equal scores stay in the order they were shot.",
            ),
        ):
            button.setCheckable(True)
            button.setObjectName("filmstripSort")
            button.setToolTip(tip)
            button.toggled.connect(lambda checked, chosen=mode: self._on_filmstrip_sort(chosen, checked))
            self._sort_group.addButton(button)
        self.sort_time.setChecked(True)
        order_label = QLabel("Frames")
        order_label.setObjectName("fieldLabel")
        strip_header = QHBoxLayout()
        strip_header.setContentsMargins(8, 6, 8, 0)
        strip_header.setSpacing(8)
        strip_header.addWidget(order_label)
        strip_header.addStretch(1)
        strip_header.addWidget(self.sort_time)
        strip_header.addWidget(self.sort_score)
        strip_box = QWidget()
        strip_box.setObjectName("filmstripFrame")
        strip_layout = QVBoxLayout(strip_box)
        strip_layout.setContentsMargins(0, 0, 0, 6)
        strip_layout.setSpacing(4)
        strip_layout.addLayout(strip_header)
        strip_layout.addWidget(self.filmstrip, 1)

        top = QSplitter(Qt.Orientation.Horizontal)
        top.addWidget(left)
        top.addWidget(self.viewer)
        top.setStretchFactor(0, 0)
        top.setStretchFactor(1, 1)
        top.setSizes([360, 920])
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(top)
        splitter.addWidget(strip_box)
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
        """
        Create the menus, toolbar, and view toggles.

        Inputs: none.
        Output: no return value. File, View, and Help actions are added. Cancel and the folder label are on the toolbar.
        Uses: the review window commands.
        Limitations: Compare, Zoom, and Focus peaking start unchecked. The folder label starts as "No folder open".
        """
        toolbar = QToolBar("Main")
        toolbar.setMovable(False)
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.addToolBar(toolbar)
        self.folder_label = QLabel("No folder open", toolbar)
        self.folder_label.setObjectName("folderLabel")
        self.folder_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.folder_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.folder_label.setMinimumWidth(180)
        self.folder_label.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.folder_label.setCursor(Qt.CursorShape.ArrowCursor)
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
        self.report_action = self._action("Save manifest…", self._export_report, "Ctrl+Shift+E")
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
        self.export_action.setToolTip(
            "Save the frames marked as keepers, and write an .xmp sidecar beside each one. "
            "Originals stay where they are."
        )
        self.report_action.setIconText("Save manifest")
        self.report_action.setToolTip(
            "Save a manifest of this folder as outburst-report.json and outburst-report.csv. "
            "It lists every frame, its score, and the keeper marks. Photos are not changed."
        )
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
            "Compare",
            "Show two frames from this burst. Starts beside the best other frame. "
            "[ and ] choose another frame. Shift-click a thumbnail to compare with it. Keyboard: C",
        )
        self.loupe_action = self._check_action(
            "Zoom",
            "Z shows the preview at 1:1. Scroll, or + and -, to zoom to 2:1 and 4:1. Drag to look around.",
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
        toolbar.addWidget(self.folder_label)

    def _action(
        self,
        text: str,
        slot,
        shortcut: str | None = None,
        icon=None,  # noqa: ANN001
    ) -> QAction:
        """
        Create a menu action that calls a slot with no arguments.

        Inputs: text, the label; slot, the callback; shortcut, a key sequence or None; icon, a toolbar icon or None.
        Output: the action, already parented to the window and connected.
        Uses: File, View, and Help commands.
        Limitations: a missing icon, or a null icon, is omitted. The triggered handler discards Qt's checked argument.
        """
        action = QAction(text, self)
        if icon is not None and not icon.isNull():
            action.setIcon(icon)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
        action.triggered.connect(lambda _checked=False, callback=slot: callback())
        return action

    def _check_action(self, text: str, tooltip: str) -> QAction:
        """
        Create a checkable view action.

        Inputs: text, the label; tooltip, the explanation.
        Output: an unchecked checkable action parented to the window.
        Uses: Compare, Zoom, and Focus peaking.
        Limitations: the action is not added to a menu here, and this method sets no shortcut.
        """
        action = QAction(text, self)
        action.setCheckable(True)
        action.setToolTip(tooltip)
        return action

    def _set_checked(self, action: QAction, checked: bool) -> None:
        """
        Set a checkable action and let its toggled handler run.

        Inputs: action, the view action; checked, the new state.
        Output: no return value. The action's checked state changes.
        Uses: the C and Z keys, and Shift-click when compare is still off.
        Limitations: signals are not blocked, so the toggled slot runs when the state changes.
        """
        action.setChecked(checked)

    def _open_folder_dialog(self) -> None:
        """
        Ask for a folder and open it.

        Inputs: none.
        Output: no return value. A chosen directory is passed to open_path.
        Uses: the Open folder action.
        Limitations: a headless process returns without a dialog. Cancelling the dialog leaves the current folder in place. The dialog starts at the open folder, or the last saved folder.
        """
        if os.environ.get("OUTBURST_HEADLESS") == "1":
            return
        start = str(self._folder or self._dialog_start)
        chosen = QFileDialog.getExistingDirectory(self, "Open a folder of bursts", start)
        if chosen:
            self.open_path(Path(chosen))

    def _start_worker(self) -> None:
        """
        Start a background scan of the open folder.

        Inputs: none. The folder, recursion, gap, weights, cache, subject model, and bird eye detection are read from the window.
        Output: no return value. The preview and filmstrip are cleared, the progress bar is shown, and the worker starts. A missing subject model or bird-eye model starts a download instead.
        Uses: opening and rescanning a folder.
        Limitations: a missing folder returns immediately. A download already in progress only remembers to scan when that download finishes. The subject model is downloaded before the bird-eye model when both are missing. When Ultralytics is missing, that library is installed before either weights file.
        """
        if self._folder is None:
            return
        self._suspend_review()
        if self._needs_subject_runtime():
            self._begin_runtime_install(then_scan=True)
            return
        if self.controls.prefer_subject() and not self._model_ready():
            self._begin_download(then_scan=True)
            return
        if self.controls.eye_scoring() and not self._bird_eye_ready():
            self._begin_bird_eye_download(then_scan=True)
            return
        if self._downloading:
            self._download_then_scan = True
            return
        self._scan_id += 1
        scan_id = self._scan_id
        subject_model = self._model_path if self.controls.prefer_subject() else None
        eye_on = self.controls.eye_scoring()
        worker = AnalyzeWorker(
            self._folder,
            recursive=self.controls.recursive(),
            threshold=self.controls.threshold(),
            weights=self.controls.weights(),
            cache_dir=self._cache_dir,
            subject_model=subject_model,
            eye_scoring=eye_on,
            eye_model=self._eye_model_path if eye_on else None,
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
        """
        Abort the scan worker and wait for it to finish.

        Inputs: none.
        Output: True when no scan is running or the worker stops within ten seconds. False when it is still running.
        Uses: opening another folder, rescanning, cancelling, and closing.
        Limitations: the scan id changes before the wait, so a late result from the old worker is ignored even when the wait fails.
        """
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
        """
        Cancel the download or the scan that is in progress.

        Inputs: none.
        Output: no return value. A download is aborted and the status says it is cancelling. A scan is stopped and the status says it was cancelled.
        Uses: Escape and the Cancel toolbar action.
        Limitations: a download is cancelled ahead of a scan. Nothing happens when neither is running.
        """
        if self._downloading and self._download_worker is not None:
            self._download_worker.abort()
            if self._download_role == "runtime":
                self.statusBar().showMessage("Cancelling install…")
            else:
                self.statusBar().showMessage("Cancelling download…")
            return
        if self._scanning:
            self._stop_worker()
            self.viewer.set_placeholder("Scan cancelled.\n\nRescan to score this folder again.")
            self.statusBar().showMessage("Scan cancelled")
            self._sync_actions()

    def _on_progress(self, done: int, total: int, message: str) -> None:
        """
        Show scan or save progress on the status bar.

        Inputs: done, the finished count; total, the expected count; message, the status text.
        Output: no return value. The progress bar range and value update, and the status bar shows the message.
        Uses: the progress bar during a scan or a save.
        Limitations: the range maximum is at least 1, and the value is clamped to that range.
        """
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(min(max(done, 0), max(total, 1)))
        self.statusBar().showMessage(message)

    def _on_success(self, scan_id: int, library: object) -> None:
        """
        Show a library from the scan that is still current.

        Inputs: scan_id, the scan this result belongs to; library, the scan result.
        Output: no return value. A matching Library is rescored with the current weights and burst gap, then presented.
        Uses: the end of a successful scan.
        Limitations: a stale scan id, or a payload that is not a Library, is ignored.
        """
        if scan_id != self._scan_id or not isinstance(library, Library):
            return
        self._scanning = False
        library = with_weights(library, self.controls.weights())
        library = with_threshold(library, self.controls.threshold())
        self.present(library)

    def _on_failed(self, scan_id: int, message: str) -> None:
        """
        Report a scan failure for the scan that is still current.

        Inputs: scan_id, the scan this result belongs to; message, the error text.
        Output: no return value. Scanning stops, actions refresh, and a warning notice is shown.
        Uses: a failed scan.
        Limitations: a stale scan id is ignored.
        """
        if scan_id != self._scan_id:
            return
        self._scanning = False
        self.viewer.set_placeholder("The scan did not finish.\n\nRescan to try this folder again.")
        self._sync_actions()
        self._notify("Scan failed", message, "warning")

    def _on_cancelled(self, scan_id: int) -> None:
        """
        Clear the scanning state after the worker reports cancellation.

        Inputs: scan_id, the scan this result belongs to.
        Output: no return value. Actions refresh and the status bar says the scan was cancelled.
        Uses: a scan that stopped because it was aborted.
        Limitations: a stale scan id is ignored.
        """
        if scan_id != self._scan_id:
            return
        self._scanning = False
        self.viewer.set_placeholder("Scan cancelled.\n\nRescan to score this folder again.")
        self._sync_actions()
        self.statusBar().showMessage("Scan cancelled")

    def _apply_threshold(self, value: float) -> None:
        """
        Regroup the open library with a new burst gap.

        Inputs: value, the gap in seconds.
        Output: no return value. Automatic keepers are cleared, bursts are rebuilt, new best frames are marked, and the list keeps the current file when it is still visible.
        Uses: the Burst gap control.
        Limitations: nothing happens before a library is loaded or while the review is cleared for a scan. This regroups measurements already in memory and does not read the files again.
        """
        if self._library is None or self._review_suspended:
            return
        selected = self._selected_path()
        self._clear_automatic_keepers()
        self._library = with_threshold(self._library, value)
        self._mark_keepers_for_new_bursts()
        self._refresh_burst_list(prefer_path=selected)

    def _apply_weights(self) -> None:
        """
        Rescore the open library with the current sharpness, exposure, center, and subject settings.

        Inputs: none. Weights are read from the control panel.
        Output: no return value. The analysis map, burst labels, filmstrip scores, and preview refresh.
        Uses: the Sharp and Center sliders, and turning subject scoring off.
        Limitations: nothing happens before a library is loaded or while the review is cleared for a scan. This rescores cached measurements and does not scan the files again. While Minimum score is on, the burst list is rebuilt because a new score can hide or show a frame.
        """
        if self._library is None or self._review_suspended:
            return
        self._library = with_weights(self._library, self.controls.weights())
        self._by_path = {item.frame.path: item for item in self._library.analyses}
        if self.controls.score_floor() is not None:
            self._refresh_burst_list(prefer_path=self._selected_path())
            return
        self._refresh_burst_labels()
        self.filmstrip.refresh_flags(self._thumbs())
        self._update_viewer()

    def _on_subject_toggled(self, enabled: bool) -> None:
        """
        Turn subject-aware sharpness on or off.

        Inputs: enabled, whether the checkbox is now checked.
        Output: no return value. Turning it off forgets a pending follow-up scan and rescores the open library. Turning it on rescans when the model is ready and a folder is open, or downloads a missing model.
        Uses: the Subject-aware sharpness checkbox.
        Limitations: with no folder open, a ready model does not start a scan. A missing Ultralytics install is installed first. A missing model downloads, and the scan follows only when a folder is open.
        """
        if not enabled:
            self._download_then_scan = False
            self._apply_weights()
            return
        if self._needs_subject_runtime():
            self._begin_runtime_install(then_scan=self._folder is not None)
            return
        if self._model_ready():
            if self._folder is not None:
                self.rescan()
            return
        self._begin_download(then_scan=self._folder is not None)

    def _on_model_changed(self, _model_id: str) -> None:
        """
        Switch to the selected subject model and turn subject scoring on.

        Inputs: _model_id, the id emitted by the combo. The argument is unused; the panel is read instead.
        Output: no return value. Subject scoring is checked without re-entering its toggle. A missing model downloads. An open folder is rescanned.
        Uses: the subject model combo.
        Limitations: with no folder, a missing model downloads and does not scan afterward. A ready model with no folder does not scan.
        """
        if not self.controls.prefer_subject():
            self.controls.set_subject_checked(True)
        if self._folder is None:
            if not self._model_ready():
                self._begin_download(then_scan=False)
            return
        self.rescan()

    def _model_ready(self) -> bool:
        """
        Remember the selected model path when its file is already in the models folder.

        Inputs: none.
        Output: True when the selected model file is ready. The stored path is set, or cleared when it is not ready.
        Uses: deciding whether a scan can score subjects without a download.
        Limitations: an unknown model id returns False and clears the stored path.
        """
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

    def _bird_eye_ready(self) -> bool:
        """
        Remember the bird-eye weights path when that file is already in the models folder.

        Inputs: none.
        Output: True when eye_pose_v0.pt is ready. The stored path is set, or cleared when it is not.
        Uses: deciding whether a scan can run bird eye detection without a download.
        Limitations: this does not download the file and does not load the network.
        """
        path = model_path(BIRD_EYE_OPTION)
        if model_is_ready(path):
            self._eye_model_path = path
            return True
        self._eye_model_path = None
        return False

    def _needs_subject_runtime(self) -> bool:
        """
        Decide whether the window should install Ultralytics before continuing.

        Inputs: none. Subject scoring, bird eye detection, and the current process are read.
        Output: True when one of those features is on, Ultralytics does not import, and this is not a headless process.
        Uses: the subject checkbox, the bird eye checkbox, and the start of a scan.
        Limitations: a headless process returns False so tests and scripted scans keep the existing stayed-off warning. A feature that is off does not request an install.
        """
        if os.environ.get("OUTBURST_HEADLESS") == "1":
            return False
        if not (self.controls.prefer_subject() or self.controls.eye_scoring()):
            return False
        return not subject_runtime_ready()

    def _begin_runtime_install(self, then_scan: bool) -> None:
        """
        Install Ultralytics, then continue the scan or model download when asked.

        Inputs: then_scan, whether a scan should start after the install and any missing weights download.
        Output: no return value. The progress bar is busy and the status says PyTorch is downloading.
        Uses: turning on subject detection or bird eye detection when the optional library is missing.
        Limitations: a download already in progress only remembers to scan afterward. Cancel stops pip the next time it is checked. Photos are not uploaded.
        """
        if self._downloading:
            self._download_then_scan = self._download_then_scan or then_scan
            return
        self._download_role = "runtime"
        self._download_then_scan = then_scan
        self._download_model_id = ""
        self._downloading = True
        worker = RuntimeInstallWorker()
        worker.succeeded.connect(self._on_runtime_installed)
        worker.failed.connect(self._on_runtime_install_failed)
        worker.cancelled.connect(self._on_runtime_install_cancelled)
        self._download_worker = worker
        self.controls.set_model_enabled(False)
        self._sync_actions()
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.statusBar().showMessage(
            "Installing subject and bird eye detection. This downloads PyTorch and can take several minutes."
        )
        worker.start()

    def _on_runtime_installed(self) -> None:
        """
        Continue after Ultralytics is installed.

        Inputs: none.
        Output: no return value. The install UI is cleared. A missing subject model or bird-eye model downloads next. Otherwise a requested scan starts.
        Uses: the end of a successful Ultralytics install.
        Limitations: the scan starts only when one was requested and a folder is open. Weights are still downloaded when the file is not ready.
        """
        follow = self._download_then_scan
        self._finish_download_ui()
        self.statusBar().showMessage("Subject and bird eye detection are installed.")
        if self.controls.prefer_subject() and not self._model_ready():
            self._begin_download(then_scan=follow)
            return
        if self.controls.eye_scoring() and not self._bird_eye_ready():
            self._begin_bird_eye_download(then_scan=follow)
            return
        if follow and self._folder is not None:
            self._start_worker()

    def _on_runtime_install_failed(self, message: str) -> None:
        """
        Leave subject and bird eye detection off when Ultralytics cannot be installed.

        Inputs: message, the pip error text.
        Output: no return value. Both features are unchecked, a warning is shown, and a requested scan continues without them.
        Uses: a failed Ultralytics install.
        Limitations: the checkboxes are changed without emitting their own scan. The following scan uses the usual sharpness score.
        """
        then_scan = self._download_then_scan
        self._finish_download_ui()
        if self.controls.eye_scoring():
            self.controls.set_eye_scoring(False)
        if self.controls.prefer_subject():
            self.controls.set_subject_checked(False)
        self.statusBar().showMessage("Subject and bird eye detection stayed off.")
        self._notify(
            "Subject detection",
            "Subject and bird eye detection could not be installed.\n\n"
            "From the Outburst folder, run:\n"
            "pip install \"outburst[subject]\"\n\n"
            f"{message}",
            "warning",
        )
        if then_scan and self._folder is not None:
            self._start_worker()

    def _on_runtime_install_cancelled(self) -> None:
        """
        Stop after the photographer cancels the Ultralytics install.

        Inputs: none.
        Output: no return value. Both features are unchecked, the status says the install was cancelled, and a requested scan continues without them.
        Uses: Escape and Cancel download during the install.
        Limitations: pip may already have written part of the library before it was stopped. The following scan uses the usual sharpness score.
        """
        then_scan = self._download_then_scan
        self._finish_download_ui()
        self.statusBar().showMessage("Subject detection install cancelled")
        if self.controls.eye_scoring():
            self.controls.set_eye_scoring(False)
        if self.controls.prefer_subject():
            self.controls.set_subject_checked(False)
        if then_scan and self._folder is not None:
            self._start_worker()

    def _begin_download(self, then_scan: bool) -> None:
        """
        Download the selected subject model, then scan when asked.

        Inputs: then_scan, whether a scan should start after the download ends.
        Output: no return value. The combo is disabled, actions refresh, and the status names the model being downloaded.
        Uses: subject scoring when the weights are not already in the models folder.
        Limitations: a headless process turns subject scoring off, says the model is missing, and scans without it when then_scan is set and a folder is open. A download of the same model only updates whether to scan afterward.
        """
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
            self._download_role = "subject"
            self._download_then_scan = then_scan
            if self._download_worker is not None:
                self._download_worker.abort()
            return
        self._download_role = "subject"
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

    def _begin_bird_eye_download(self, then_scan: bool) -> None:
        """
        Download the bird-eye pose weights, then scan when asked.

        Inputs: then_scan, whether a scan should start after the download ends.
        Output: no return value. The model combo is disabled and the status names Bird eye detection.
        Uses: turning Bird eye detection on when eye_pose_v0.pt is not in the models folder.
        Limitations: a headless process turns bird eye detection off, says the model is missing, and scans without it when then_scan is set and a folder is open. Photos are not uploaded. The weights are AGPL-3.0, separate from Outburst.
        """
        if os.environ.get("OUTBURST_HEADLESS") == "1":
            self.controls.set_eye_scoring(False)
            self.statusBar().showMessage("Bird eye model is not in the models folder")
            if then_scan and self._folder is not None:
                self._start_worker()
            return
        option = BIRD_EYE_OPTION
        if self._downloading:
            if self._download_model_id == option.id:
                self._download_then_scan = self._download_then_scan or then_scan
                return
            self._replace_download = True
            self._download_role = "bird-eye"
            self._download_then_scan = then_scan
            if self._download_worker is not None:
                self._download_worker.abort()
            return
        self._download_role = "bird-eye"
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
        """
        Show how much of the subject model has downloaded.

        Inputs: done, bytes received; total, the expected size, or 0 when the size is unknown.
        Output: no return value. The progress bar updates and the status shows the received size in megabytes.
        Uses: the status bar during a model download.
        Limitations: a total of 0 or less leaves the progress bar busy with no endpoint.
        """
        if total <= 0:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, total)
            self.progress.setValue(min(done, total))
        self.statusBar().showMessage(f"Downloading model… {done / (1024 * 1024):.1f} MB")

    def _on_download_succeeded(self, path: str) -> None:
        """
        Store a downloaded model and scan when that was requested.

        Inputs: path, the saved model file.
        Output: no return value. The download UI is cleared, the model list refreshes, and a requested scan starts when subject scoring is still on and a folder is open.
        Uses: the end of a subject-model download.
        Limitations: without a following scan, the status names the saved file and no scan starts.
        """
        then_scan = self._download_then_scan
        role = self._download_role
        self._finish_download_ui()
        if role == "bird-eye":
            self._eye_model_path = Path(path)
        else:
            self._model_path = Path(path)
            self.controls.refresh_models()
        wants_scan = then_scan and self._folder is not None
        if role == "bird-eye":
            wants_scan = wants_scan and self.controls.eye_scoring()
        else:
            wants_scan = wants_scan and self.controls.prefer_subject()
        if wants_scan:
            self._start_worker()
            return
        self.statusBar().showMessage(f"Saved {Path(path).name} in the models folder")

    def _on_download_failed(self, message: str) -> None:
        """
        Report a failed model download and continue with a scan when one was requested.

        Inputs: message, the error text.
        Output: no return value. A warning is shown, subject scoring is turned off, and a requested scan starts when a folder is open.
        Uses: a subject-model download that failed.
        Limitations: subject scoring is cleared only when it is still checked. The scan then runs without the model.
        """
        then_scan = self._download_then_scan
        role = self._download_role
        self._finish_download_ui()
        self._notify("Could not download the model", message, "warning")
        if role == "bird-eye":
            if self.controls.eye_scoring():
                self.controls.set_eye_scoring(False)
        elif self.controls.prefer_subject():
            self.controls.set_subject_checked(False)
        if then_scan and self._folder is not None:
            self._start_worker()

    def _on_download_cancelled(self) -> None:
        """
        Finish a cancelled download, or start the model the photographer picked instead.

        Inputs: none.
        Output: no return value. A replacement download starts when one was requested. Otherwise the status says the download was cancelled, subject scoring is turned off, and a requested scan starts.
        Uses: cancelling a subject-model download.
        Limitations: a replacement download returns before the cancelled status is shown.
        """
        replace = self._replace_download
        role = self._download_role
        then_scan = self._download_then_scan
        self._replace_download = False
        self._finish_download_ui()
        if replace:
            if role == "bird-eye":
                self._begin_bird_eye_download(then_scan)
            else:
                self._begin_download(then_scan)
            return
        self.statusBar().showMessage("Model download cancelled")
        if role == "bird-eye":
            if self.controls.eye_scoring():
                self.controls.set_eye_scoring(False)
        elif self.controls.prefer_subject():
            self.controls.set_subject_checked(False)
        if then_scan and self._folder is not None:
            self._start_worker()

    def _finish_download_ui(self) -> None:
        """
        Clear the download flags and enable the model combo again.

        Inputs: none.
        Output: no return value. The worker reference is dropped, a still-running worker is given two seconds, and actions refresh.
        Uses: every end of a subject-model download.
        Limitations: the flag that remembers a following scan is cleared here, so the caller must read it first.
        """
        worker = self._download_worker
        self._downloading = False
        self._download_then_scan = False
        self._download_worker = None
        if worker is not None and worker.isRunning():
            worker.wait(2000)
        self.controls.set_model_enabled(True)
        self._sync_actions()

    def _on_minimum_changed(self, _value: int) -> None:
        """
        Hide or show bursts after Minimum frames changes.

        Inputs: _value, the new minimum. The argument is unused; the control panel is read while the list is rebuilt.
        Output: no return value. The burst list refreshes and keeps the current file when it is still visible. An open library also refreshes the status summary.
        Uses: the Minimum control.
        Limitations: bursts shorter than the minimum stay off the list and out of the filmstrip.
        """
        self._refresh_burst_list(prefer_path=self._selected_path())
        if self._library is not None:
            self.statusBar().showMessage(self._status_summary())

    def _on_minimum_score_changed(self) -> None:
        """
        Hide or show frames after Minimum score changes.

        Inputs: none. The checkbox and cutoff are read from the control panel.
        Output: no return value. The burst list and filmstrip refresh and keep the current file when it still meets the cutoff. An open library also refreshes the status summary.
        Uses: the Minimum score checkbox and its number.
        Limitations: with the checkbox off, every frame stays visible. A frame with no score stays hidden while the checkbox is on. Marks on a hidden frame are kept.
        """
        self._refresh_burst_list(prefer_path=self._selected_path())
        if self._library is not None:
            self.statusBar().showMessage(self._status_summary())

    def _best_visible_index(self, burst: Burst) -> int | None:
        """
        Find the highest-scoring frame that meets Minimum score.

        Inputs: burst, the burst to search.
        Output: a shooting-order index, or None when no frame meets the cutoff.
        Uses: opening a burst whose recommended frame is hidden, and the recommended-frame shortcut.
        Limitations: with Minimum score off, every frame is eligible and the highest composite wins. A missing score sorts last.
        """
        visible = visible_frame_indexes(
            burst.frames,
            self._by_path,
            minimum_score=self.controls.score_floor(),
        )
        if not visible:
            return None

        def rank(index: int) -> float:
            analysis = self._by_path.get(burst.frames[index].path)
            if analysis is None or analysis.composite is None:
                return -1.0
            return analysis.composite

        return max(visible, key=rank)

    def _nothing_visible_message(self) -> str:
        """
        Explain why the burst list is empty after a scan.

        Inputs: none. Minimum frames, Minimum score, and the open library are read from the window.
        Output: placeholder text. A folder whose long bursts all fall under Minimum score names that cutoff. Otherwise the text names Minimum frames.
        Uses: the preview when a scanned folder has no visible burst.
        Limitations: the text describes the controls. It does not change them. An eye too small to rate is still counted.
        """
        library = self._library
        if library is None:
            return _WELCOME
        floor = self.controls.score_floor()
        long_enough = [burst for burst in library.bursts if burst.count >= self.controls.minimum_frames()]
        eyes = detected_eye_count(library.analyses) if library.eye_active else 0
        if floor is not None and long_enough:
            text = (
                f"Every burst still long enough to review scores under {floor}.\n\n"
                "Lower Minimum score, or turn it off, to see those frames."
            )
            short = len(library.bursts) - len(long_enough)
            if short == 1:
                text += "\n\n1 shorter burst stays hidden by Minimum frames."
            elif short > 1:
                text += f"\n\n{short} shorter bursts stay hidden by Minimum frames."
            if eyes == 1:
                text += "\n\nBird eye detection found an eye on 1 frame."
            elif eyes > 1:
                text += f"\n\nBird eye detection found an eye on {eyes} frames."
            return text
        return _short_bursts_hidden(self.controls.minimum_frames(), eyes=eyes)

    def _visible_bursts(self) -> tuple[Burst, ...]:
        """
        Return the bursts long enough to review.

        Inputs: none.
        Output: the bursts that meet Minimum frames and still have a frame at the minimum score, in library order, or an empty tuple before a library is loaded.
        Uses: the burst list, filmstrip, and status summary.
        Limitations: shorter bursts stay hidden. While Minimum score is on, a burst whose frames all fall below that score is not returned.
        """
        if self._library is None:
            return ()
        return visible_bursts(
            self._library.bursts,
            minimum_frames=self.controls.minimum_frames(),
            by_path=self._by_path,
            minimum_score=self.controls.score_floor(),
        )

    def _status_summary(self) -> str:
        """
        Summarize the open library for the status bar.

        Inputs: none.
        Output: an empty string before a library is loaded. Otherwise file and visible-burst counts, bursts hidden by length, frames and bursts hidden by Minimum score, note count, bird eye detection and how many eyes were found, and compare, zoom, or peaking when those are on.
        Uses: the status bar.
        Limitations: hidden bursts are counted and not named. A zoom of 1 is shown as 1:1. Any other zoom is shown as an integer ratio. Frames hidden by Minimum score are counted inside bursts that meet Minimum frames.
        """
        library = self._library
        if library is None:
            return ""
        shown = len(self._visible_bursts())
        long_enough = [burst for burst in library.bursts if burst.count >= self.controls.minimum_frames()]
        short_hidden = len(library.bursts) - len(long_enough)
        noun = "burst" if shown == 1 else "bursts"
        message = f"{len(library.analyses)} files · {shown} {noun}"
        if short_hidden:
            hidden_noun = "burst" if short_hidden == 1 else "bursts"
            message += (
                f" · {short_hidden} {hidden_noun} under {self.controls.minimum_frames()} frames hidden"
            )
        floor = self.controls.score_floor()
        if floor is not None:
            hidden_frames = 0
            hidden_bursts = 0
            for burst in long_enough:
                kept = len(visible_frame_indexes(burst.frames, self._by_path, minimum_score=floor))
                if kept == 0:
                    hidden_bursts += 1
                else:
                    hidden_frames += burst.count - kept
            if hidden_frames:
                frame_noun = "frame" if hidden_frames == 1 else "frames"
                message += f" · {hidden_frames} {frame_noun} under {floor} hidden"
            if hidden_bursts:
                burst_noun = "burst" if hidden_bursts == 1 else "bursts"
                message += f" · {hidden_bursts} {burst_noun} under {floor} hidden"
        if library.warnings:
            message += f" · {len(library.warnings)} notes"
        modes: list[str] = []
        if self._compare:
            modes.append("compare")
        if self._zoom_scale == 1:
            modes.append("1:1")
        elif self._zoom_scale is not None:
            modes.append(f"{int(self._zoom_scale)}:1")
        if self._peaking:
            modes.append("focus peaking")
        if library.eye_active:
            eyes = detected_eye_count(library.analyses)
            noun = "bird eye" if eyes == 1 else "bird eyes"
            message += f" · bird eye detection · {eyes} {noun}"
        if modes:
            message += " · " + " · ".join(modes)
        return message

    def _on_burst_row(self, row: int) -> None:
        """
        Show the burst the photographer selected in the list.

        Inputs: row, the list row.
        Output: no return value. A row inside the visible bursts shows that burst, starting at its recommended frame.
        Uses: the burst list.
        Limitations: a negative row, or a row past the visible bursts, is ignored.
        """
        if row < 0 or row >= len(self._visible_bursts()):
            return
        self._show_burst(row, prefer_path=None)

    def _update_score_effect(self) -> None:
        """
        Show how many frames Minimum score is leaving on screen.

        Inputs: none.
        Output: no return value. The Scoring row reads "N of M showing" while the filter is on. The line is hidden when the filter is off or no library is open.
        Uses: the Minimum score control after the burst list is rebuilt.
        Limitations: M counts frames in bursts that meet Minimum frames. N counts the ones still on a filmstrip. Short bursts are left out of both numbers. The line does not change marks or Save keepers.
        """
        library = self._library
        floor = self.controls.score_floor()
        if library is None or floor is None:
            self.controls.set_score_effect(None, None)
            return
        minimum = self.controls.minimum_frames()
        eligible = 0
        showing = 0
        for burst in library.bursts:
            if burst.count < minimum:
                continue
            eligible += burst.count
            showing += len(visible_frame_indexes(burst.frames, self._by_path, minimum_score=floor))
        self.controls.set_score_effect(showing, eligible)

    def _refresh_burst_list(self, prefer_path: Path | None) -> None:
        """
        Rebuild the burst list from the visible bursts.

        Inputs: prefer_path, a file to keep selected, or None to select the first visible burst.
        Output: no return value. The list, filmstrip, and preview update. No visible bursts shows the short-burst message when the library has bursts, and the welcome message otherwise.
        Uses: the burst list after a scan, a gap change, a minimum change, or a minimum-score change.
        Limitations: bursts under Minimum frames stay off the list. Bursts whose frames all fall below Minimum score stay off the list while that filter is on. List signals are blocked while the rows are rebuilt. Nothing is drawn while the review is cleared for a scan.
        """
        if self._review_suspended:
            return
        library = self._library
        self._update_score_effect()
        self.burst_list.blockSignals(True)
        self.burst_list.clear()
        if library is None:
            self.burst_list.blockSignals(False)
            self.viewer.set_placeholder(_WELCOME)
            return
        floor = self.controls.score_floor()
        for burst in self._visible_bursts():
            shown = len(visible_frame_indexes(burst.frames, self._by_path, minimum_score=floor))
            item = QListWidgetItem(
                burst_list_text(
                    burst,
                    self._by_path,
                    self._decisions,
                    shown_frames=None if floor is None else shown,
                )
            )
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
            self.viewer.set_placeholder(self._nothing_visible_message())
        else:
            self.viewer.set_placeholder(_WELCOME)

    def _refresh_burst_labels(self) -> None:
        """
        Rewrite the burst-list text without changing which row is selected.

        Inputs: none.
        Output: no return value. Each visible row's text is updated from the current scores and decisions.
        Uses: the burst list after a mark or a rescore.
        Limitations: nothing happens before a library is loaded. Rows are not added or removed.
        """
        if self._library is None:
            return
        floor = self.controls.score_floor()
        for row, burst in enumerate(self._visible_bursts()):
            item = self.burst_list.item(row)
            if item is not None:
                shown = len(visible_frame_indexes(burst.frames, self._by_path, minimum_score=floor))
                item.setText(
                    burst_list_text(
                        burst,
                        self._by_path,
                        self._decisions,
                        shown_frames=None if floor is None else shown,
                    )
                )

    def _show_burst(self, index: int, prefer_path: Path | None) -> None:
        """
        Show one visible burst and choose which frame is current.

        Inputs: index, the visible-burst index; prefer_path, a file to select, or None for the recommended frame.
        Output: no return value. The compare partner is cleared, the filmstrip is rebuilt, and the preview updates.
        Uses: moving between bursts.
        Limitations: an index outside the visible bursts returns immediately. A preferred path that is hidden or not in the burst falls back to the recommended frame when that frame is visible, and otherwise to the highest visible score. A scan in progress returns without drawing.
        """
        if self._review_suspended:
            return
        bursts = self._visible_bursts()
        if not (0 <= index < len(bursts)):
            return
        self._burst_index = index
        burst = bursts[index]
        self._compare_index = None
        visible = set(visible_frame_indexes(burst.frames, self._by_path, minimum_score=self.controls.score_floor()))
        frame_index = None
        if prefer_path is not None:
            found = next((i for i, frame in enumerate(burst.frames) if frame.path == prefer_path), None)
            if found in visible:
                frame_index = found
        if frame_index is None and burst.frames:
            recommended = recommended_offset(burst.frames, self._by_path)
            frame_index = recommended if recommended in visible else self._best_visible_index(burst)
        self._frame_index = frame_index or 0
        self.filmstrip.set_cards(self._thumbs())
        self._update_viewer()

    def _select_frame(self, index: int) -> None:
        """
        Make one frame in the current burst the current frame.

        Inputs: index, the shooting-order frame index.
        Output: no return value. The index is clamped, the filmstrip selection moves, and the preview updates.
        Uses: the filmstrip, the arrow keys, Home and End, and the recommended-frame shortcut.
        Limitations: a burst with no frames returns immediately. When the compare partner was this frame, the partner moves to another frame in on-screen order.
        """
        burst = self._current_burst()
        if burst is None or not burst.frames:
            return
        index = max(0, min(index, len(burst.frames) - 1))
        visible = self._filmstrip_indexes()
        if visible and index not in visible:
            index = min(visible, key=lambda item: (abs(item - index), item))
        self._frame_index = index
        if self._compare_index == self._frame_index:
            self._compare_index = self._adjacent_frame(self._frame_index, 1)
        self.filmstrip.mark_selected(self._frame_index)
        self._update_viewer()

    def _step_frame(self, delta: int) -> None:
        """
        Move to the next or previous frame in on-screen order.

        Inputs: delta, a positive step toward the end of the filmstrip or a negative step toward the start.
        Output: no return value. The frame at the neighboring filmstrip position becomes current.
        Uses: Right and Left.
        Limitations: arrow keys follow the on-screen order. The step stops at the first and last card. An empty filmstrip does nothing. A current index missing from the order starts from the first card.
        """
        order = self._filmstrip_indexes()
        if not order:
            return
        try:
            position = order.index(self._frame_index)
        except ValueError:
            position = 0
        position = max(0, min(position + delta, len(order) - 1))
        self._select_frame(order[position])

    def _on_filmstrip_sort(self, mode: str, checked: bool) -> None:
        """
        Switch the filmstrip between shooting order and score order.

        Inputs: mode, "time" or "score"; checked, whether that button is now checked.
        Output: no return value. A newly checked mode is stored, and an open burst rebuilds the filmstrip and preview.
        Uses: the Time and Score buttons above the filmstrip.
        Limitations: an unchecked signal, or the mode that is already active, is ignored. Time is shooting order and Score puts the highest composite first with ties in shooting order.
        """
        if not checked or mode == self._filmstrip_sort:
            return
        self._filmstrip_sort = mode
        if self._review_suspended or self._current_burst() is None:
            return
        self.filmstrip.set_cards(self._thumbs())
        self._update_viewer()

    def _filmstrip_indexes(self) -> list[int]:
        """
        List frame indexes in the current filmstrip order.

        Inputs: none.
        Output: shooting-order indexes, or score order. An empty list when no burst is current.
        Uses: arrow keys, Home and End, compare movement, and the filmstrip cards.
        Limitations: Time is shooting order and Score puts the highest composite first with ties in shooting order. A frame with no analysis contributes no score and stays at the right in score order. While Minimum score is on, a frame below that score is left out.
        """
        burst = self._current_burst()
        if burst is None:
            return []
        scores: list[float | None] = []
        for frame in burst.frames:
            analysis = self._by_path.get(frame.path)
            scores.append(None if analysis is None else analysis.composite)
        visible = set(
            visible_frame_indexes(burst.frames, self._by_path, minimum_score=self.controls.score_floor())
        )
        return [index for index in filmstrip_order(scores, self._filmstrip_sort) if index in visible]

    def _step_burst(self, delta: int) -> None:
        """
        Move to the next or previous visible burst.

        Inputs: delta, a positive step toward the end of the list or a negative step toward the start.
        Output: no return value. The burst list selection moves, which shows that burst.
        Uses: Down and Up.
        Limitations: the step stops at the first and last visible burst. An empty list does nothing.
        """
        count = len(self._visible_bursts())
        if count == 0:
            return
        row = max(0, min(self._burst_index + delta, count - 1))
        self.burst_list.setCurrentRow(row)

    def _go_to_recommended(self) -> None:
        """
        Select the recommended frame in the current burst.

        Inputs: none.
        Output: no return value. The recommended frame becomes current.
        Uses: the Enter shortcut.
        Limitations: a missing burst, or a burst with no frames, does nothing. When the recommended frame is below Minimum score, the highest visible score is selected instead.
        """
        burst = self._current_burst()
        if burst is None or not burst.frames:
            return
        recommended = recommended_offset(burst.frames, self._by_path)
        if recommended in self._filmstrip_indexes():
            self._select_frame(recommended)
            return
        best = self._best_visible_index(burst)
        if best is not None:
            self._select_frame(best)

    def _toggle_decision(self, kind: str) -> None:
        """
        Toggle the current frame between a mark and no mark.

        Inputs: kind, "keep" or "reject".
        Output: no return value. The same mark clears. A different mark replaces it. The list, filmstrip, and preview refresh.
        Uses: the keeper and reject buttons and the A and X keys.
        Limitations: nothing happens while a scan or a save is running, or when no frame is current. The mark is stored as a photographer mark. Original files are not modified.
        """
        analysis = self._current_analysis()
        if analysis is None or self._review_locked():
            return
        path = analysis.frame.path
        updated = None if self._decisions.get(path) == kind else kind
        self._apply_decision(path, updated, origin="user")
        self._refresh_burst_labels()
        self.filmstrip.refresh_flags(self._thumbs())
        self._update_viewer()

    def _reject_burst(self) -> None:
        """
        Reject every frame in the current burst, or clear those rejects.

        Inputs: none.
        Output: no return value. When every frame is already rejected, the rejects are cleared. Otherwise every frame is rejected. The list, filmstrip, and preview refresh.
        Uses: the R key.
        Limitations: nothing happens while a scan or a save is running, or when no burst is current. Original files are not modified.
        """
        burst = self._current_burst()
        if burst is None or self._review_locked():
            return
        clear = all(self._decisions.get(frame.path) == "reject" for frame in burst.frames)
        for frame in burst.frames:
            self._apply_decision(frame.path, None if clear else "reject", origin="user")
        self._refresh_burst_labels()
        self.filmstrip.refresh_flags(self._thumbs())
        self._update_viewer()

    def _on_compare_toggled(self, checked: bool) -> None:
        """
        Show or hide the second preview.

        Inputs: checked, whether Compare is on.
        Output: no return value. The preview updates and the status summary is refreshed.
        Uses: the Compare action and the C key.
        Limitations: the partner frame is chosen when the preview is redrawn. This method only stores the flag and refreshes.
        """
        self._compare = checked
        self._update_viewer()
        self._show_mode_status()

    def _on_loupe_toggled(self, checked: bool) -> None:
        """
        Turn the 1:1 preview on or off.

        Inputs: checked, whether Zoom is on.
        Output: no return value. Checked sets the scale to 1. Unchecked returns to fit. Both panes and the status summary update.
        Uses: the Zoom action and the Z key.
        Limitations: this toggle is only fit or 1:1. The 2:1 and 4:1 steps come from the wheel or the plus and minus keys. Zoom uses the embedded preview.
        """
        self._zoom_scale = 1.0 if checked else None
        self._loupe = checked
        self.viewer.set_view_mode(scale=self._zoom_scale, peak=self._peaking)
        self._show_mode_status()

    def _on_peak_toggled(self, checked: bool) -> None:
        """
        Turn focus peaking on or off.

        Inputs: checked, whether the strongest edges should be painted green.
        Output: no return value. Both previews redraw and the status summary updates.
        Uses: the Focus peaking action and the F key.
        Limitations: peaking uses the embedded preview. The current zoom scale is left as it is.
        """
        self._peaking = checked
        self.viewer.set_view_mode(scale=self._zoom_scale, peak=self._peaking)
        self._show_mode_status()

    def _on_subject_box_toggled(self, _checked: bool) -> None:
        """
        Show or hide the detected animal box.

        Inputs: _checked, whether the box is drawn. The checkbox is read again while the preview is redrawn.
        Output: no return value. The preview and compare pane redraw.
        Uses: the Show subject box checkbox in the Scoring section.
        Limitations: hiding the box does not turn off subject scoring and does not rescan. A frame with no detected animal still has no box. Zoom still centers on the animal when the box is hidden and no eye is present.
        """
        self._update_viewer()

    def _on_eye_circle_toggled(self, _checked: bool) -> None:
        """
        Show or hide the bird eye ring.

        Inputs: _checked, whether the ring is drawn. The checkbox is read again while the preview is redrawn.
        Output: no return value. The preview and compare pane redraw.
        Uses: the Show eye circle checkbox in the Scoring section.
        Limitations: hiding the ring does not turn off bird eye detection and does not rescan. The info line, filmstrip, and ranking still use the eye. Zoom still centers on the eye when the ring is hidden.
        """
        self._update_viewer()

    def _on_zoom_changed(self, scale: float) -> None:
        """
        Remember a zoom change that came from the preview.

        Inputs: scale, the new scale, or 0 when the preview returned to fit.
        Output: no return value. The stored scale and the Zoom action match the preview, without toggling zoom again, and the status summary updates.
        Uses: the scroll wheel over the preview.
        Limitations: a scale of 0 or below is stored as fit. Zoom steps are fit, 1, 2, and 4 and use the embedded preview.
        """
        self._zoom_scale = None if scale <= 0 else scale
        self._loupe = self._zoom_scale is not None
        self.loupe_action.blockSignals(True)
        self.loupe_action.setChecked(self._loupe)
        self.loupe_action.blockSignals(False)
        self._show_mode_status()

    def _step_zoom(self, steps: int) -> None:
        """
        Move the preview one zoom step.

        Inputs: steps, a positive step to zoom in or a negative step to zoom out.
        Output: no return value. The scale becomes the next of fit, 1:1, 2:1, and 4:1. Both panes, the Zoom action, and the status summary update.
        Uses: the plus and minus keys.
        Limitations: zoom steps are fit, 1, 2, and 4 and use the embedded preview. This path does not keep a cursor anchor.
        """
        self._zoom_scale = next_zoom(self._zoom_scale, steps)
        self._loupe = self._zoom_scale is not None
        self.loupe_action.blockSignals(True)
        self.loupe_action.setChecked(self._loupe)
        self.loupe_action.blockSignals(False)
        self.viewer.set_view_mode(scale=self._zoom_scale, peak=self._peaking)
        self._show_mode_status()

    def _show_mode_status(self) -> None:
        """
        Refresh the status bar after compare, zoom, or peaking changes.

        Inputs: none.
        Output: no return value. When a library is open, the status bar shows the library summary.
        Uses: the status bar.
        Limitations: nothing is written before a library is loaded.
        """
        if self._library is not None:
            self.statusBar().showMessage(self._status_summary())

    def _select_partner(self) -> None:
        """
        Make the compare frame the current frame and keep the previous frame as the partner.

        Inputs: none.
        Output: no return value. The partner becomes the old current frame when that frame is different, and the partner frame becomes current.
        Uses: a click on the compare preview.
        Limitations: a missing burst, or no other frame, does nothing. When compare has not chosen a partner, the partner starts as the best other frame.
        """
        burst = self._current_burst()
        if burst is None:
            return
        partner = self._compare_index
        if partner is None or partner == self._frame_index:
            partner = compare_partner_offset(burst.frames, self._by_path, self._frame_index)
        if partner is None:
            return
        previous = self._frame_index
        self._compare_index = previous if previous != partner else self._adjacent_frame(partner, 1)
        self._select_frame(partner)

    def _set_compare_frame(self, index: int) -> None:
        """
        Choose the other side of compare.

        Inputs: index, a shooting-order frame index in the current burst.
        Output: no return value. The partner index is stored. Compare is turned on when it was off. The status asks for a different frame when the index is the current frame or out of range.
        Uses: Shift-click on a thumbnail and the bracket keys.
        Limitations: a burst with fewer than two frames does nothing. Turning compare on returns through the Compare toggle, which redraws the preview.
        """
        burst = self._current_burst()
        if burst is None or not burst.frames or len(burst.frames) < 2:
            return
        if index == self._frame_index or not 0 <= index < len(burst.frames):
            self.statusBar().showMessage("Choose a different frame to compare.")
            return
        self._compare_index = index
        if not self._compare:
            self._set_checked(self.compare_action, True)
            return
        self._update_viewer()
        self._show_mode_status()

    def _step_compare(self, delta: int) -> None:
        """
        Move the compare partner along the filmstrip.

        Inputs: delta, a positive or negative direction. The sign chooses the direction.
        Output: no return value. The partner becomes another frame in on-screen order.
        Uses: the [ and ] keys.
        Limitations: the step follows the on-screen order and wraps. A burst with fewer than two frames, or no other frame, does nothing. When compare is off or has no partner, the step starts from the best other frame.
        """
        burst = self._current_burst()
        if burst is None or len(burst.frames) < 2:
            return
        start = self._compare_index
        if start is None or not self._compare or start == self._frame_index:
            start = compare_partner_offset(burst.frames, self._by_path, self._frame_index)
        if start is None:
            return
        chosen = self._adjacent_frame(start, delta)
        if chosen is None:
            return
        self._set_compare_frame(chosen)

    def _adjacent_frame(self, start: int, delta: int) -> int | None:
        """
        Find another frame in on-screen order, skipping the current frame.

        Inputs: start, a shooting-order index; delta, the direction. The sign of delta chooses the way around the filmstrip.
        Output: a shooting-order index, or None when the filmstrip has fewer than two frames, delta is 0, or every other candidate is the current frame.
        Uses: moving the compare partner and keeping the partner different from the current frame.
        Limitations: the search follows the on-screen order and wraps. A start index that is not on the filmstrip begins at the first card.
        """
        order = self._filmstrip_indexes()
        if len(order) < 2 or delta == 0:
            return None
        try:
            position = order.index(start)
        except ValueError:
            position = 0
        step = 1 if delta > 0 else -1
        for offset in range(1, len(order)):
            candidate = order[(position + step * offset) % len(order)]
            if candidate != self._frame_index:
                return candidate
        return None

    def _reject_obvious(self) -> None:
        """
        Reject soft or badly clipped frames, leaving the best frame and your keepers. Pressing again restores the frames this action rejected in the same burst.

        Inputs: none.
        Output: no return value. Obvious misses are rejected, or the remembered rejects are cleared when they are all still rejected. The list, filmstrip, preview, and status update.
        Uses: the G key.
        Limitations: nothing happens while a scan or a save is running, or when no burst or library is current. A burst with no obvious misses only updates the status bar. Original files are not modified.
        """

        burst = self._current_burst()
        if burst is None or self._review_locked() or self._library is None:
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
        """
        Redraw the preview, marks, and compare pane for the current frame.

        Inputs: none.
        Output: no return value. The preview shows the frame, its info, border, badges, and stars. Compare shows the partner, or a one-frame message when there is no other frame.
        Uses: the large preview.
        Limitations: zoom uses the embedded preview. With no current frame the welcome placeholder is shown. While a scan has cleared the review, the scanning message stays up. A rejected frame uses the red border ahead of the keeper and recommended colors. The subject box is drawn only when subject scoring is active, a box was found, and Subject box is checked. The eye ring is drawn only when an eye was stored and Eye circle is checked. Hiding either mark leaves the zoom center on that point.
        """
        if self._review_suspended:
            self.viewer.set_placeholder(_SCANNING_PREVIEW)
            return
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
        circle = self._eye_circle(analysis)
        summary, detail = frame_info_parts(
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
            box if self.controls.show_subject_box() else None,
            summary,
            border,
            keeper=keeper,
            rejected=rejected,
            stars=self._stars.get(frame.path),
            circle=circle if self.controls.show_eye_circle() else None,
            focus=_overlay_focus(box, circle),
            detail=detail,
        )
        self.viewer.set_view_mode(scale=self._zoom_scale, peak=self._peaking)
        self.viewer.set_compare(self._compare)
        if not self._compare:
            self.filmstrip.mark_compared(None)
            return
        partner_index = self._compare_index
        if partner_index is None or partner_index == self._frame_index or not 0 <= partner_index < len(burst.frames):
            partner_index = compare_partner_offset(burst.frames, self._by_path, self._frame_index)
        if partner_index not in self._filmstrip_indexes():
            partner_index = self._adjacent_frame(self._frame_index, 1)
        self.filmstrip.mark_compared(partner_index)
        if partner_index is None:
            if burst.count > 1 and self.controls.score_floor() is not None:
                self.viewer.set_partner_message("Compare", "No other frame meets the minimum score.")
            else:
                self.viewer.set_partner_message("Compare", "This burst has one frame.")
            return
        partner_frame = burst.frames[partner_index]
        partner = self._by_path.get(partner_frame.path)
        partner_decision = self._decisions.get(partner_frame.path)
        partner_box = None
        partner_circle = None if partner is None else self._eye_circle(partner)
        if (
            show_subject
            and partner is not None
            and partner.metrics is not None
        ):
            partner_box = partner.metrics.subject_box
        if partner_index == recommended_index:
            role = "Best in the burst"
        elif self._compare_index is None:
            role = "Next sharpest"
        else:
            role = f"Frame {partner_index + 1} of {burst.count}"
        score = score_caption(
            None if partner is None else partner.composite,
            None if partner is None else partner.metrics,
            best_composite=None,
        )
        self.viewer.set_partner(
            None if partner is None else partner.preview_path,
            partner_box if self.controls.show_subject_box() else None,
            f"{role}  ·  {score}  ·  {partner_frame.path.name}",
            circle=partner_circle if self.controls.show_eye_circle() else None,
            focus=_overlay_focus(partner_box, partner_circle),
            keeper=partner_decision == "keep",
            rejected=partner_decision == "reject",
        )

    def _thumbs(self) -> list[Thumb]:
        """
        Build the filmstrip cards for the current burst.

        Inputs: none.
        Output: thumbnail descriptions in filmstrip order, or an empty list when no burst or library is current.
        Uses: the filmstrip.
        Limitations: Time is shooting order and Score puts the highest composite first with ties in shooting order. A filename longer than 22 characters is shortened in the card title. The tooltip keeps the full name.
        """
        burst = self._current_burst()
        if burst is None or self._library is None:
            return []
        best = recommended_offset(burst.frames, self._by_path) if burst.frames else 0
        best_analysis = self._by_path.get(burst.frames[best].path) if burst.frames else None
        best_composite = None if best_analysis is None else best_analysis.composite
        cards: dict[int, Thumb] = {}
        for index, frame in enumerate(burst.frames):
            analysis = self._by_path.get(frame.path)
            decision = self._decisions.get(frame.path)
            name = frame.path.name
            title = name if len(name) <= 22 else f"{name[:9]}…{name[-10:]}"
            cards[index] = Thumb(
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
                compared=self._compare and index == self._compare_index,
                best=index == best,
                decision=decision,
                tooltip=name,
                frame_index=index,
            )
        return [cards[index] for index in self._filmstrip_indexes()]

    def _export_keepers(self) -> None:
        """
        Ask where to save marked keepers and start that save.

        Inputs: none.
        Output: no return value. A chosen folder starts a copy or hard-link save and writes an .xmp beside each saved keeper. No marked keepers in bursts that meet the minimum shows a notice.
        Uses: the Save keepers action.
        Limitations: bursts under Minimum frames are left out. A cancelled dialog, or a headless process, does nothing. Original files are not modified. The sidecar is written in the keepers folder.
        """
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
        self._start_save(
            ExportWorker(
                [choice.path for choice in choices],
                request.folder,
                request.mode,
                self._library,
                dict(self._decisions),
                dict(self._stars),
            ),
            f"Saving {len(choices)} keeper(s)…",
        )

    def _export_report(self) -> None:
        """
        Write a JSON manifest and a CSV manifest for the open library into a chosen folder.

        Inputs: none.
        Output: no return value. The status bar names the two files. A write error shows a warning.
        Uses: the Save manifest action.
        Limitations: a headless process, or no library, returns without a dialog. Cancelling the dialog does nothing. The report is given the current stars and Minimum frames.
        """
        if self._library is None or os.environ.get("OUTBURST_HEADLESS") == "1":
            return
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Save a manifest",
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
            self._notify("Could not save the manifest", str(exc), "warning")
            return
        self.statusBar().showMessage(f"Wrote {json_path.name} and {csv_path.name}")

    def _write_xmp(self) -> None:
        """
        Confirm, then write XMP sidecars for keepers and star-rated frames.

        Inputs: none.
        Output: no return value. After confirmation, a background write starts. No matching frames shows a notice.
        Uses: the Write XMP sidecars action.
        Limitations: sidecars are keepers and star-rated frames only. headless ask() returns False, so the write does not start. Original files are not modified.
        """
        if self._library is None:
            return
        count = len(
            frames_for_sidecars(
                self._library,
                self._decisions,
                stars=self._stars,
                minimum_frames=self.controls.minimum_frames(),
            )
        )
        if count == 0:
            self._notify(
                "No sidecars to write",
                "Mark a keeper or set a star rating first. A starred frame is written even when it is not a keeper.",
                "info",
            )
            return
        noun = "file" if count == 1 else "files"
        if not ask(
            self,
            "Write XMP sidecars",
            f"Write sidecars next to {count} {noun}?\n\n"
            "Outburst writes one sidecar for each keeper, and for each frame with a star rating, "
            "including a frame you rated and left unpicked. Other frames are left alone. "
            "Original RAW and JPEG files stay unchanged.",
        ):
            return
        self._start_save(
            SidecarWorker(
                self._library,
                dict(self._decisions),
                dict(self._stars),
                self.controls.minimum_frames(),
            ),
            f"Writing sidecars for {count} {noun}…",
        )

    def _show_warnings(self) -> None:
        """
        Open the scan-notes dialog for the warnings stored on the window.

        Inputs: none.
        Output: no return value. The stored warnings are passed to the dialog helper.
        Uses: the Scan notes action.
        Limitations: an empty warning list, or a headless process, opens nothing.
        """
        show_warnings(self, self._warnings)

    def _open_cache_folder(self) -> None:
        """
        Open the cache directory in the system file browser.

        Inputs: none.
        Output: no return value. The desktop service is asked to open the cache path.
        Uses: the Open cache folder action.
        Limitations: this asks the operating system to open the path. It does not create the folder.
        """
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._cache_dir)))

    def _open_current_folder(self) -> None:
        """
        Open the folder under review in the system file browser.

        Inputs: none.
        Output: no return value. The desktop service is asked to open the folder. Nothing happens when no folder is open.
        Uses: a click on the toolbar folder path.
        Limitations: this asks the operating system to open the path. Rescan stays on the toolbar and is not started by the click.
        """
        if self._folder is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._folder)))

    def _clear_cache(self) -> None:
        """
        Delete cached measurements and previews after confirmation, then rescan an open folder.

        Inputs: none.
        Output: no return value. On confirmation the measurement cache is cleared, the status counts the removed files, and an open folder is rescanned.
        Uses: the Clear cache action.
        Limitations: headless ask() returns False, so nothing is deleted. A scan, download, or save in progress shows a warning and returns. Keeper, reject, and star marks stay.
        """
        if self._scanning or self._downloading or self._saving:
            self._notify("Still working", "Wait until the current save or scan finishes before clearing the cache.", "warning")
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
        """
        Delete cached measurements for one folder.

        Inputs: folder, the folder whose cached measurements should be dropped.
        Output: the number of forgotten photograph paths, or 0 when the cache cannot be cleared.
        Uses: Rescan, so the next scan measures that folder again.
        Limitations: a cache error shows a warning and returns 0. Keeper, reject, and star marks stay.
        """
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
        """
        Mark the recommended frame when a burst does not already have a keeper.

        When two or more frames have a rated eye, the sharpest eye is the frame Outburst marks. Earlier automatic keepers in that burst are cleared. A reject stays rejected, and a keeper the photographer set keeps its origin.

        Inputs: none.
        Output: no return value. A burst with no keeper gains an automatic keeper on the recommended frame, except when that frame is already rejected.
        Uses: showing a library after a scan.
        Limitations: nothing happens before a library is loaded. A rejected recommended frame is skipped. Original files are not modified.
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
        """
        Rescan after bird eye detection is turned on or off.

        Inputs: enabled, whether the checkbox is now checked.
        Output: no return value. Automatic keepers are cleared. Turning it off rescans. Turning it on rescans when the weights are ready, or downloads them first.
        Uses: the Bird eye detection checkbox.
        Limitations: with no folder open, a missing model still downloads, and no scan follows. A ready model with no folder does not scan. A missing Ultralytics install is installed before the weights download.
        """
        self._clear_automatic_keepers()
        if not enabled:
            if self._folder is not None:
                self.rescan()
            return
        if self._needs_subject_runtime():
            self._begin_runtime_install(then_scan=self._folder is not None)
            return
        if self._bird_eye_ready():
            if self._folder is not None:
                self.rescan()
            return
        self._begin_bird_eye_download(then_scan=self._folder is not None)

    def _eye_circle(self, analysis: FrameAnalysis) -> tuple[int, int, int] | None:
        """
        Return the eye circle to draw on a preview.

        Inputs: analysis, the frame measurement.
        Output: center x, center y, and radius as integers, or None when bird eye detection is off or this frame has no eye center or radius.
        Uses: the eye mark on the preview and the compare pane.
        Limitations: the circle is in embedded-preview pixels. A missing library, or a library scanned without bird eye detection, returns None.
        """
        if self._library is None or not self._library.eye_active:
            return None
        metrics = analysis.metrics
        if metrics is None or metrics.eye_center is None or metrics.eye_radius is None:
            return None
        x, y = metrics.eye_center
        return (int(x), int(y), int(metrics.eye_radius))

    def _clear_automatic_keepers(self) -> None:
        """
        Drop keepers Outburst chose. Marks the photographer set stay.

        Inputs: none.
        Output: no return value. Keep marks whose origin is automatic are cleared.
        Uses: a burst-gap change and turning bird eye detection on or off.
        Limitations: a photographer's keep or reject is left in place. Original files are not modified.
        """

        automatic = [
            path
            for path, origin in self._origins.items()
            if origin == "auto" and self._decisions.get(path) == "keep"
        ]
        for path in automatic:
            self._apply_decision(path, None)

    def _mark_keepers_for_new_bursts(self) -> None:
        """
        Mark each burst's best frame after the burst gap changes.

        A frame the photographer rejected stays rejected. A frame they already marked as a keeper keeps that mark. Every other best frame is marked automatically.

        Inputs: none.
        Output: no return value. The recommended frame of each burst is stored as an automatic keeper unless it was rejected or the photographer already kept it.
        Uses: the Burst gap control.
        Limitations: nothing happens before a library is loaded. Every burst in the library is considered, including bursts hidden by Minimum frames. Original files are not modified.
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
        """
        Show the About Outburst message.

        Inputs: none.
        Output: no return value. A message box names the version and says the photographs stay on this computer.
        Uses: the Help menu.
        Limitations: a headless process returns without a message box. The text says original files are not modified.
        """
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
        """
        Return the burst selected in the list.

        Inputs: none.
        Output: the visible burst at the current index, or None when the list is empty or the index is out of range.
        Uses: the preview, filmstrip, and keyboard navigation.
        Limitations: bursts under Minimum frames are not returned.
        """
        bursts = self._visible_bursts()
        if not bursts or not (0 <= self._burst_index < len(bursts)):
            return None
        return bursts[self._burst_index]

    def _current_analysis(self) -> FrameAnalysis | None:
        """
        Return the measurement for the current frame.

        Inputs: none.
        Output: the frame analysis, or None when no frame is current or the path has no measurement.
        Uses: the preview and the keeper, reject, and star actions.
        Limitations: the index is clamped to the burst before the path is read.
        """
        burst = self._current_burst()
        if burst is None or not burst.frames:
            return None
        index = max(0, min(self._frame_index, len(burst.frames) - 1))
        return self._by_path.get(burst.frames[index].path)

    def _selected_path(self) -> Path | None:
        """
        Return the file path of the current frame.

        Inputs: none.
        Output: the photograph path, or None when no frame analysis is current.
        Uses: keeping the same file selected after the burst list is rebuilt.
        Limitations: a current frame with no analysis returns None.
        """
        analysis = self._current_analysis()
        if analysis is None:
            return None
        return analysis.frame.path

    def _row_for_path(self, path: Path | None) -> int:
        """
        Find the burst-list row that contains a file.

        Inputs: path, a photograph path, or None.
        Output: the visible-burst index, or 0 when no library is loaded, the path is None, or the file is not in a visible burst.
        Uses: restoring the selection after Minimum frames or the burst gap changes.
        Limitations: a burst hidden by Minimum frames is not searched, so a file in that burst returns 0.
        """
        if self._library is None or path is None:
            return 0
        for index, burst in enumerate(self._visible_bursts()):
            if any(frame.path == path for frame in burst.frames):
                return index
        return 0

    def _load_decisions(self) -> None:
        """
        Load saved keeper, reject, and star marks for the open library.

        Inputs: none.
        Output: no return value. The in-memory decisions, origins, and stars are replaced from the cache.
        Uses: showing a library.
        Limitations: a cache error logs a warning and clears the in-memory marks. No library clears them as well.
        """
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
        """
        Store one keep or reject mark and write it to the cache.

        Inputs: path, the photograph; decision, "keep", "reject", or None to clear; origin, "user" or "auto", used for a keep.
        Output: no return value. The in-memory maps update and the cache is asked to save the mark.
        Uses: keeper and reject actions, including automatic keepers.
        Limitations: a reject is stored with user origin. Clearing a mark drops both the decision and its origin. Original files are not modified.
        """
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
        """
        Set or clear the star rating on the current frame.

        Inputs: stars, 0 to clear, or 1 through 5.
        Output: no return value. The same rating again clears it. The rating is saved, and the list, filmstrip, and preview refresh.
        Uses: the star buttons and the 0 through 5 keys.
        Limitations: nothing happens while a scan or a save is running, or when no frame is current. The rating is stored in the cache. The photograph file is not modified.
        """
        analysis = self._current_analysis()
        if analysis is None or self._review_locked():
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
        """
        Write one star rating to the cache.

        Inputs: path, the photograph; stars, 1 through 5, or None to clear.
        Output: no return value. A cache error shows a warning.
        Uses: the star rating on the current frame.
        Limitations: the photograph file is not modified.
        """
        try:
            with Cache(self._cache_dir) as cache:
                cache.set_stars(path, stars)
        except OSError as exc:
            self._notify("Could not save that rating", str(exc), "warning")

    def _save_decision(self, path: Path, decision: str | None, *, origin: str = "user") -> None:
        """
        Write one keep or reject mark to the cache.

        Inputs: path, the photograph; decision, "keep", "reject", or None; origin, "user" or "auto".
        Output: no return value. A cache error shows a warning.
        Uses: keeper and reject marks.
        Limitations: original files are not modified. A cache error leaves the in-memory mark in place and shows a warning.
        """
        try:
            with Cache(self._cache_dir) as cache:
                cache.set_decision(path, decision, origin=origin)
        except OSError as exc:
            self._notify("Could not save that decision", str(exc), "warning")

    def _set_folder_label(self) -> None:
        """
        Show the open folder path on the toolbar.

        Inputs: none.
        Output: no return value. The label shows the folder path and a hand cursor, or the text "No folder open".
        Uses: the toolbar folder label.
        Limitations: a missing folder clears the tooltip and uses an arrow cursor. The click opens the folder and does not rescan it.
        """
        if self._folder is None:
            self.folder_label.setText("No folder open")
            self.folder_label.setToolTip("")
            self.folder_label.setCursor(Qt.CursorShape.ArrowCursor)
            self.folder_label.setProperty("open", "false")
        else:
            self.folder_label.setText(str(self._folder))
            self.folder_label.setToolTip("Open this folder")
            self.folder_label.setCursor(Qt.CursorShape.PointingHandCursor)
            self.folder_label.setProperty("open", "true")
        style = self.folder_label.style()
        style.unpolish(self.folder_label)
        style.polish(self.folder_label)

    def _suspend_review(self) -> None:
        """
        Clear the burst list, preview, and filmstrip for a scan.

        Inputs: none.
        Output: no return value. The list and filmstrip are emptied, the preview shows the scanning message, and those controls are disabled.
        Uses: Rescan and the start of a folder scan.
        Limitations: keeper marks stay in memory. The cleared frames are not drawn again until a finished library is presented. Arrow keys and filmstrip clicks do nothing while this is in effect. The controls stay enabled so focus does not jump to a slider.
        """
        self._review_suspended = True
        self.burst_list.blockSignals(True)
        self.burst_list.clear()
        self.burst_list.blockSignals(False)
        self.filmstrip.set_cards([])
        self.viewer.set_placeholder(_SCANNING_PREVIEW)
        self.controls.set_score_effect(None, None)

    def _resume_review(self) -> None:
        """
        Allow the burst list, preview, and filmstrip to show a finished library.

        Inputs: none.
        Output: no return value. The cleared review can be drawn again.
        Uses: presenting a library after a scan.
        Limitations: this does not draw frames. The caller rebuilds the list and preview.
        """
        self._review_suspended = False

    def _review_locked(self) -> bool:
        """
        Report whether keeper, reject, and star changes should wait.

        Inputs: none.
        Output: True while a scan or a save is running, or while the review is cleared for a scan.
        Uses: the mark and rating actions.
        Limitations: a model download locks review only when the preview was already cleared for the scan that follows.
        """
        return self._scanning or self._saving or self._review_suspended

    def _start_save(self, worker: QThread, message: str) -> None:
        """
        Run a keeper save or a sidecar write on a background thread.

        Inputs: worker, the export or sidecar thread; message, the status text shown while it runs.
        Output: no return value. The progress bar resets, the status shows the message, and the worker starts.
        Uses: Save keepers and Write XMP sidecars.
        Limitations: a save already in progress is ignored, including its worker. This method does not abort the worker.
        """
        if self._saving:
            return
        self._save_id += 1
        save_id = self._save_id
        self._save_worker = worker
        self._saving = True
        worker.progress.connect(self._on_progress)  # type: ignore[attr-defined]
        worker.succeeded.connect(lambda payload, ident=save_id: self._on_save_finished(ident, payload))  # type: ignore[attr-defined]
        worker.failed.connect(lambda message, ident=save_id: self._on_save_failed(ident, str(message)))  # type: ignore[attr-defined]
        self._sync_actions()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.statusBar().showMessage(message)
        worker.start()

    def _on_save_finished(self, save_id: int, payload: object) -> None:
        """
        Report a finished keeper save or sidecar write.

        Inputs: save_id, the save this result belongs to; payload, a sidecar tuple of written paths and warnings, or a list of export results.
        Output: no return value. Saving stops and the status bar summarizes what was written, including the sidecar count for a keeper save. Sidecar warnings replace the stored scan notes.
        Uses: the end of Save keepers or Write XMP sidecars.
        Limitations: a stale save id is ignored. A payload that is neither a tuple nor a list is summarized as zero keepers.
        """
        if save_id != self._save_id:
            return
        self._saving = False
        self._save_worker = None
        if isinstance(payload, tuple):
            written, warnings = payload
            message = f"Wrote {len(written)} sidecar(s)."
            if warnings:
                message += " " + str(warnings[0])
                self._warnings = tuple(str(item) for item in warnings)
        else:
            results = payload if isinstance(payload, list) else []
            notes = [getattr(result, "note", "") for result in results if getattr(result, "note", "")]
            notes.extend(
                getattr(result, "sidecar_note", "") for result in results if getattr(result, "sidecar_note", "")
            )
            sidecars = [result for result in results if getattr(result, "sidecar", None) is not None]
            folder = results[0].destination.parent if results else ""
            message = f"Saved {len(results)} keeper(s)"
            if folder:
                message += f" to {folder}"
            message += f" and wrote {len(sidecars)} sidecar(s)"
            if notes:
                message += " " + str(notes[0])
        self._sync_actions()
        self.statusBar().showMessage(message)

    def _on_save_failed(self, save_id: int, message: str) -> None:
        """
        Report a failed keeper save or sidecar write.

        Inputs: save_id, the save this result belongs to; message, the error text.
        Output: no return value. Saving stops, actions refresh, and a warning notice is shown.
        Uses: a failed Save keepers or Write XMP sidecars action.
        Limitations: a stale save id is ignored.
        """
        if save_id != self._save_id:
            return
        self._saving = False
        self._save_worker = None
        self._sync_actions()
        self._notify("Save failed", message, "warning")

    def _sync_actions(self) -> None:
        """
        Enable the actions that match the current scan, download, and save state.

        Inputs: none.
        Output: no return value. Save, report, and sidecar actions need a library and no scan or save. Cancel is shown during a scan or a download. The progress bar is shown during a scan, download, or save.
        Uses: the toolbar and menus.
        Limitations: Open is disabled only while saving. Rescan needs an open folder and no scan, download, or save. Save actions stay available during a model download when a library is already loaded.
        """
        busy = self._scanning or self._downloading or self._saving
        ready = self._library is not None and not self._scanning and not self._saving
        self.export_action.setEnabled(ready)
        self.report_action.setEnabled(ready)
        self.xmp_action.setEnabled(ready)
        self.open_action.setEnabled(not self._saving)
        self.rescan_action.setEnabled(self._folder is not None and not busy)
        self.cancel_action.setEnabled(self._scanning or self._downloading)
        self.cancel_action.setVisible(self._scanning or self._downloading)
        if self._downloading and self._download_role == "runtime":
            self.cancel_action.setText("Cancel install")
        elif self._downloading:
            self.cancel_action.setText("Cancel download")
        else:
            self.cancel_action.setText("Cancel scan")
        self.warnings_action.setEnabled(bool(self._warnings))
        self.progress.setVisible(busy)

    def _set_filmstrip_sort(self, mode: str) -> None:
        """
        Set the filmstrip order without handling the button toggle.

        Inputs: mode, "score", or any other string, which selects Time.
        Output: no return value. The stored mode updates and the matching button is checked.
        Uses: restoring the saved filmstrip order when the window opens.
        Limitations: button signals are blocked, so the filmstrip is not rebuilt here. Time is shooting order and Score puts the highest composite first with ties in shooting order.
        """
        chosen = "score" if mode == "score" else "time"
        self._filmstrip_sort = chosen
        button = self.sort_score if chosen == "score" else self.sort_time
        for item in (self.sort_time, self.sort_score):
            item.blockSignals(True)
        button.setChecked(True)
        for item in (self.sort_time, self.sort_score):
            item.blockSignals(False)

    def _notify(self, title: str, text: str, level: str) -> None:
        """
        Show a notice through the shared dialog helper.

        Inputs: title, the notice title; text, the message; level, "warning" or "info".
        Output: no return value. The helper logs the notice and shows a message box when one can be shown.
        Uses: warnings and information from the review window.
        Limitations: a headless process logs the notice and does not open a message box.
        """
        notify(self, title, text, level)

    def _load_settings(self) -> None:
        """
        Restore the controls and the last folder from saved settings.

        Inputs: none.
        Output: no return value. Burst gap, sharpness, center, subfolders, minimum, minimum score, subject, eye, subject box, eye circle, the Info panel, filmstrip order, Grouping, Scoring, and model id are applied. A saved folder becomes the next dialog start.
        Uses: opening the review window.
        Limitations: the saved folder is not scanned. A missing minimum uses 1 when an older show-single-bursts setting is true, and 2 otherwise.
        """
        settings = QSettings("Outburst", "Outburst")
        self.controls.set_threshold(_as_float(settings.value("threshold"), 1.0))
        self.controls.set_sharpness_percent(_as_int(settings.value("sharpness"), 75))
        self.controls.set_center_percent(_as_int(settings.value("center_mix"), 65))
        self.controls.set_recursive(_as_bool(settings.value("recursive"), False))
        self.controls.set_minimum_frames(_saved_minimum_frames(settings))
        self.controls.set_minimum_score(_as_int(settings.value("minimum_score"), 50))
        self.controls.set_minimum_score_enabled(_as_bool(settings.value("minimum_score_enabled"), False))
        self.controls.set_subject_checked(_as_bool(settings.value("prefer_subject"), False))
        self.controls.set_eye_scoring(_as_bool(settings.value("eye_scoring"), False))
        self.controls.set_show_subject_box(_as_bool(settings.value("show_subject_box"), True))
        self.controls.set_show_eye_circle(_as_bool(settings.value("show_eye_circle"), True))
        self.viewer.set_info_open(_as_bool(settings.value("show_frame_detail"), False))
        self.controls.set_grouping_open(_as_bool(settings.value("grouping_open"), False))
        self.controls.set_scoring_open(_as_bool(settings.value("scoring_open"), True))
        self._set_filmstrip_sort(str(settings.value("filmstrip_sort", "time") or "time"))
        folder = settings.value("last_folder", "")
        if folder:
            self._dialog_start = Path(str(folder))
        self.controls.set_model_id(_saved_model_id(settings))
        self._model_ready()

    def _save_settings(self) -> None:
        """
        Store the controls and the open folder for the next launch.

        Inputs: none.
        Output: no return value. Burst gap, sharpness, center, subfolders, minimum, minimum score, subject, eye, subject box, eye circle, the Info panel, filmstrip order, Grouping, Scoring, and model id are written.
        Uses: closing the review window.
        Limitations: the folder is saved only when one is open. The resolved model path is saved only when one is stored.
        """
        settings = QSettings("Outburst", "Outburst")
        settings.setValue("threshold", self.controls.threshold())
        settings.setValue("sharpness", self.controls.sharpness_percent())
        settings.setValue("center_mix", self.controls.center_percent())
        settings.setValue("recursive", self.controls.recursive())
        settings.setValue("minimum_frames", self.controls.minimum_frames())
        settings.setValue("minimum_score", self.controls.minimum_score())
        settings.setValue("minimum_score_enabled", self.controls.minimum_score_enabled())
        settings.setValue("prefer_subject", self.controls.prefer_subject())
        settings.setValue("eye_scoring", self.controls.eye_scoring())
        settings.setValue("show_subject_box", self.controls.show_subject_box())
        settings.setValue("show_eye_circle", self.controls.show_eye_circle())
        settings.setValue("show_frame_detail", self.viewer.info_open())
        settings.setValue("grouping_open", self.controls.grouping_open())
        settings.setValue("scoring_open", self.controls.scoring_open())
        settings.setValue("filmstrip_sort", self._filmstrip_sort)
        settings.setValue("subject_model_id", self.controls.model_id())
        if self._folder is not None:
            settings.setValue("last_folder", str(self._folder))
        if self._model_path is not None:
            settings.setValue("subject_model", str(self._model_path))


def _overlay_focus(
    box: tuple[int, int, int, int] | None,
    circle: tuple[int, int, int] | None,
) -> tuple[int, int] | None:
    """
    Choose the zoom center for a frame's detected marks.

    Inputs: box, the subject rectangle or None; circle, the eye center, radius, or None.
    Output: the eye center when a circle is present, otherwise the center of the box, or None when both are missing.
    Uses: the preview and compare pane so hiding a mark still leaves zoom on the animal or the eye.
    Limitations: the box center uses integer division. A circle of any radius uses its center.
    """
    if circle is not None:
        return (circle[0], circle[1])
    if box is None:
        return None
    x, y, width, height = box
    return (x + width // 2, y + height // 2)


def _typing_target(widget: QWidget | None) -> bool:
    """
    Report whether the focused widget should keep the key for itself.

    Inputs: widget, the focused widget, or None.
    Output: True for a line edit, spin box, slider, or combo box.
    Uses: the keyboard filter, so review shortcuts do not take those keys.
    Limitations: keys are ignored while a spin box is focused. The same is true for a slider, a line edit, and a combo box.
    """
    return isinstance(widget, (QLineEdit, QAbstractSpinBox, QAbstractSlider, QComboBox))


def _as_float(value: object, default: float) -> float:
    """
    Read a saved setting as a float.

    Inputs: value, the stored object; default, the value used when conversion fails.
    Output: the float, or default.
    Uses: restoring the burst gap.
    Limitations: a value that cannot be converted, including None, returns default.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int) -> int:
    """
    Read a saved setting as an integer.

    Inputs: value, the stored object; default, the value used when conversion fails.
    Output: the integer, after converting through float, or default.
    Uses: restoring sharpness, center, and minimum frames.
    Limitations: a non-numeric value returns default. A fractional value is truncated toward zero.
    """
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _saved_minimum_frames(settings: QSettings) -> int:
    """
    Read the saved minimum burst length, including the older single-burst setting.

    Inputs: settings, the application settings.
    Output: at least 1. A stored minimum is used. When none was stored, a true older show-single-bursts setting returns 1, and otherwise the result is 2.
    Uses: restoring Minimum frames.
    Limitations: a stored value below 1 is raised to 1. The older setting is used only when minimum_frames was never stored.
    """
    stored = settings.value("minimum_frames")
    if stored not in (None, ""):
        return max(1, _as_int(stored, 2))
    if _as_bool(settings.value("show_single_bursts"), False):
        return 1
    return 2


def _saved_model_id(settings: QSettings) -> str:
    """
    Read the saved subject model id, including an older saved path.

    Inputs: settings, the application settings.
    Output: a known model id, or the default model id.
    Uses: restoring the subject model combo.
    Limitations: an unknown id falls back to a model whose filename matches the old stored path, and then to the default id.
    """
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
    """
    Read a saved setting as a boolean.

    Inputs: value, the stored object; default, the value used when the setting is missing.
    Output: a real bool is returned as itself. None returns default. Otherwise the strings "1", "true", and "yes", in any case, are True.
    Uses: restoring checkboxes.
    Limitations: any other string, including "false", returns False rather than default.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in {"1", "true", "yes"}
