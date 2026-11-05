"""Small dialogs and message boxes for the review window."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from outburst.gui.shortcuts import shortcuts_html

log = logging.getLogger(__name__)


def headless() -> bool:
    """
    Report whether dialogs should stay off the screen.

    Inputs: none.
    Output: True when the OUTBURST_HEADLESS environment variable is the string "1".
    Uses: every review-window dialog that would otherwise open.
    Limitations: any other value, including an unset variable, is treated as an interactive session.
    """
    return os.environ.get("OUTBURST_HEADLESS") == "1"


def notify(parent: QWidget | None, title: str, text: str, level: str = "info") -> None:
    """
    Log a notice and, when a window is available, show it in a message box.

    Inputs: parent, the widget that owns the box; title and text, the message; level, "warning" or another level treated as information.
    Output: no return value. The message is logged, stored on parent.last_notice when that attribute exists, and shown in a message box.
    Uses: scan, save, and cache notices in the review window.
    Limitations: a headless process, or a missing parent, logs the notice and does not open a message box.
    """
    if level == "warning":
        log.warning("%s: %s", title, text)
    else:
        log.info("%s: %s", title, text)
    if parent is not None and hasattr(parent, "last_notice"):
        parent.last_notice = (title, text)  # type: ignore[attr-defined]
    if headless() or parent is None:
        return
    box = QMessageBox(parent)
    box.setWindowTitle(title)
    box.setText(text)
    box.setIcon(QMessageBox.Icon.Warning if level == "warning" else QMessageBox.Icon.Information)
    box.exec()


def ask(parent: QWidget | None, title: str, text: str) -> bool:
    """
    Ask a yes-or-no question before a write or a cache clear.

    Inputs: parent, the widget that owns the box; title and text, the question.
    Output: True only when the photographer clicks Yes.
    Uses: cache clearing and the XMP sidecar confirmation.
    Limitations: headless ask() returns False.
    """
    if headless():
        return False
    answer = QMessageBox.question(parent, title, text)
    return answer == QMessageBox.StandardButton.Yes


@dataclass(frozen=True)
class ExportRequest:
    folder: Path
    mode: str


class ExportKeepersDialog(QDialog):
    def __init__(self, parent: QWidget | None, start: Path | None) -> None:
        """
        Build the save-keepers dialog with a folder field and copy or hard-link choices.

        Inputs: parent, the owning widget; start, the folder shown in the field, or None for the home folder.
        Output: no return value. The dialog is titled, filled, and sized.
        Uses: the Save keepers dialog.
        Limitations: copy is selected by default. The dialog does not create the folder, and the note says original files are not modified.
        """
        super().__init__(parent)
        self.setWindowTitle("Save keepers")
        self._start = start or Path.home()
        self._folder = QLineEdit(str(self._start))
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse)
        self._copy = QRadioButton("Copy files")
        self._link = QRadioButton("Hard link when possible")
        self._copy.setChecked(True)
        self._copy.setToolTip("A full copy in the keepers folder. The original stays where it is.")
        self._link.setToolTip(
            "A second name for the same file, so this uses almost no extra disk space. "
            "If a link cannot be created, Outburst copies the file instead."
        )
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        group = QButtonGroup(self)
        group.addButton(self._copy)
        group.addButton(self._link)

        layout = QVBoxLayout(self)
        summary = QLabel(
            "Outburst saves marked keepers from bursts that meet Minimum frames. "
            "Shorter bursts stay hidden and are not saved."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        row = QHBoxLayout()
        row.addWidget(self._folder, 1)
        row.addWidget(browse)
        layout.addLayout(row)
        layout.addWidget(self._copy)
        layout.addWidget(self._link)
        note = QLabel("Originals are not modified. A file already in that folder is left as it is.")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addWidget(buttons)
        self.resize(520, 220)

    @staticmethod
    def ask(parent: QWidget | None, start: Path | None) -> ExportRequest | None:
        """
        Open the save-keepers dialog and return the chosen folder and mode.

        Inputs: parent, the owning widget; start, the folder initially shown.
        Output: an ExportRequest with the folder and "hardlink" or "copy", or None when the dialog is cancelled or the folder text is empty.
        Uses: the Save keepers action.
        Limitations: a headless process returns None without showing the dialog. Original files are not modified.
        """
        if headless():
            return None
        dialog = ExportKeepersDialog(parent, start)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        folder = Path(dialog._folder.text().strip())
        if not str(folder):
            return None
        mode = "hardlink" if dialog._link.isChecked() else "copy"
        return ExportRequest(folder, mode)

    def _browse(self) -> None:
        """
        Let the photographer pick the keepers folder.

        Inputs: none.
        Output: no return value. A chosen directory replaces the folder field text.
        Uses: the Browse button on the Save keepers dialog.
        Limitations: cancelling the directory dialog leaves the field unchanged.
        """
        chosen = QFileDialog.getExistingDirectory(self, "Keepers folder", self._folder.text() or str(self._start))
        if chosen:
            self._folder.setText(chosen)


def show_shortcuts(parent: QWidget | None) -> None:
    """
    Show the keyboard-shortcut list in a dialog.

    Inputs: parent, the widget that owns the dialog.
    Output: no return value. The dialog stays open until it is closed.
    Uses: the Help menu keyboard shortcuts action.
    Limitations: a headless process returns without opening the dialog.
    """
    if headless():
        return
    dialog = QDialog(parent)
    dialog.setWindowTitle("Keyboard shortcuts")
    browser = QTextBrowser()
    browser.setHtml(shortcuts_html())
    browser.setOpenExternalLinks(False)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    buttons.accepted.connect(dialog.accept)
    layout = QVBoxLayout(dialog)
    layout.addWidget(browser)
    layout.addWidget(buttons)
    dialog.resize(560, 420)
    dialog.exec()


def show_warnings(parent: QWidget | None, warnings: tuple[str, ...]) -> None:
    """
    Show scan notes in a dialog.

    Inputs: parent, the widget that owns the dialog; warnings, the note strings.
    Output: no return value. Each note is listed as a bullet after HTML escaping.
    Uses: the View menu Scan notes action.
    Limitations: a headless process, or an empty warning tuple, returns without opening the dialog.
    """
    if headless() or not warnings:
        return
    dialog = QDialog(parent)
    dialog.setWindowTitle("Scan notes")
    browser = QTextBrowser()
    items = "".join(f"<li>{_escape(item)}</li>" for item in warnings)
    browser.setHtml(f"<p>Outburst finished the scan with these notes.</p><ul>{items}</ul>")
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout = QVBoxLayout(dialog)
    layout.addWidget(browser)
    layout.addWidget(buttons)
    dialog.resize(560, 320)
    dialog.exec()


def _escape(text: str) -> str:
    """
    Escape the HTML characters used in scan notes.

    Inputs: text, one warning string.
    Output: the same text with ampersands, less-than signs, and greater-than signs replaced by entities.
    Uses: the scan-notes dialog.
    Limitations: quotes and other characters are left unchanged.
    """
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
