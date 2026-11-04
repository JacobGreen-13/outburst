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
    return os.environ.get("OUTBURST_HEADLESS") == "1"


def notify(parent: QWidget | None, title: str, text: str, level: str = "info") -> None:
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
        chosen = QFileDialog.getExistingDirectory(self, "Keepers folder", self._folder.text() or str(self._start))
        if chosen:
            self._folder.setText(chosen)


def show_shortcuts(parent: QWidget | None) -> None:
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
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
