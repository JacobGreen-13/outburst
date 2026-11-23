"""Dark review theme. A dim surround makes it easier to judge a photograph.

``apply_theme`` sets the Fusion palette and the stylesheet for the rail, preview, and filmstrip. ``build_app_icon`` loads the committed Outburst icon.
"""

from __future__ import annotations

import sys
from importlib.resources import as_file, files
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QImage, QPalette, QPixmap
from PySide6.QtWidgets import QApplication

STYLESHEET = """
QWidget {
    color: #f2f2f2;
    background: #161616;
    font-size: 13px;
}
QMainWindow, QMenuBar, QMenu, QStatusBar, QToolBar {
    background: #161616;
    color: #f2f2f2;
}
QMenuBar::item:selected, QMenu::item:selected {
    background: #1f3d30;
}
QToolBar {
    border: none;
    spacing: 6px;
    padding: 4px;
}
QLabel#title {
    font-size: 22px;
    font-weight: 650;
    background: transparent;
}
QLabel#subtitle, QLabel#folderLabel, QLabel#hint, QLabel#fieldLabel {
    color: #a3a3a3;
    background: transparent;
}
QLabel#folderLabel {
    padding-right: 8px;
}
QLabel#folderLabel[open="true"] {
    color: #d7d7d7;
}
QLabel#folderLabel[open="true"]:hover {
    color: #ffffff;
    text-decoration: underline;
}
QToolButton#sectionToggle {
    background: transparent;
    border: none;
    border-radius: 4px;
    color: #cfcfcf;
    font-weight: 600;
    padding: 2px 4px;
    text-align: left;
}
QToolButton#sectionToggle:hover {
    background: #2a2a2a;
}
QWidget#sectionBody {
    background: #1c1c1c;
    border: 1px solid #333333;
    border-radius: 8px;
}
QWidget#sidePanel {
    background: transparent;
}
QWidget#filmstripFrame {
    background: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 8px;
}
QScrollArea#filmstrip {
    background: transparent;
    border: none;
}
QListWidget {
    background: #1e1e1e;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    border-radius: 6px;
    padding: 6px;
    margin: 2px;
}
QListWidget::item:selected {
    background: #1f3d30;
    color: #f4fff8;
}
QScrollArea {
    background: #1a1a1a;
    border: 1px solid #333333;
    border-radius: 8px;
}
QGroupBox {
    border: 1px solid #333333;
    border-radius: 8px;
    margin-top: 12px;
    padding: 8px;
    background: #1c1c1c;
}
QGroupBox#sectionBox {
    margin-top: 14px;
    padding: 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #cfcfcf;
}
QPushButton {
    background: #2a2a2a;
    border: 1px solid #3d3d3d;
    border-radius: 6px;
    padding: 6px 10px;
}
QPushButton:hover {
    background: #333333;
}
QPushButton:pressed {
    background: #1f3d30;
}
QPushButton:disabled {
    color: #777777;
}
QPushButton#keeperButton, QPushButton#rejectButton {
    min-height: 36px;
    font-size: 14px;
    font-weight: 600;
    padding: 8px 16px;
}
QPushButton#keeperButton:checked {
    background: #1f6b45;
    color: #f4fff8;
    border: 1px solid #7dffc3;
}
QPushButton#rejectButton:checked {
    background: #6b2a2a;
    color: #fff5f5;
    border: 1px solid #e06c6c;
}
QPushButton#filmstripSort {
    padding: 3px 12px;
    min-height: 22px;
}
QPushButton#filmstripSort:checked {
    background: #1f3d30;
    color: #f4fff8;
    border: 1px solid #7dffc3;
}
QSlider::groove:horizontal {
    height: 4px;
    background: #3a3a3a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 14px;
    margin: -6px 0;
    background: #3dbe86;
    border-radius: 7px;
}
QDoubleSpinBox, QSpinBox, QLineEdit, QComboBox {
    background: #242424;
    color: #f2f2f2;
    border: 1px solid #3a3a3a;
    border-radius: 6px;
    padding: 4px 8px;
}
QComboBox::drop-down {
    border: none;
    width: 22px;
}
QComboBox QAbstractItemView {
    background: #242424;
    color: #f2f2f2;
    selection-background-color: #1f3d30;
    selection-color: #f2f2f2;
    border: 1px solid #3a3a3a;
}
QCheckBox {
    background: transparent;
    spacing: 6px;
}
QProgressBar {
    border: 1px solid #333333;
    border-radius: 4px;
    background: #242424;
    text-align: center;
    color: #f2f2f2;
    min-width: 140px;
    max-height: 14px;
}
QProgressBar::chunk {
    background: #1f8a5b;
    border-radius: 3px;
}
QSplitter::handle:horizontal {
    width: 8px;
    background: #161616;
}
QSplitter::handle:vertical {
    height: 8px;
    background: #161616;
}
QStatusBar {
    color: #cfcfcf;
}
QToolTip {
    color: #f2f2f2;
    background: #222222;
    border: 1px solid #3a3a3a;
}
QLabel#infoPanel {
    background: #1b1b1b;
    padding: 8px 10px 10px 10px;
}
"""


def apply_theme(app: QApplication) -> None:
    """
    Apply the dark Fusion palette and stylesheet to the application.

    Inputs: app, the Qt application.
    Output: no return value. The style, palette, and stylesheet are set on app.
    Uses: the review window theme.
    Limitations: rules in the stylesheet override the palette for the widgets they name.
    """
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#161616"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#f2f2f2"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#1e1e1e"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#242424"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#f2f2f2"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#2a2a2a"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#f2f2f2"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#1f8a5b"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#222222"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#f2f2f2"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#8d8d8d"))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


def _icon_image() -> QImage:
    """
    Read the committed Outburst icon.

    Inputs: none. The file is outburst.png beside this module.
    Output: the icon image, or a null image when the file is missing or unreadable.
    Uses: the window icon and the Windows .ico writer.
    Limitations: the pixels come from the PNG. A missing file does not draw a replacement.
    """

    candidates = [Path(__file__).resolve().with_name("outburst.png")]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "outburst" / "gui" / "outburst.png")
    for path in candidates:
        if path.is_file():
            image = QImage(str(path))
            if not image.isNull():
                return image
    try:
        resource = files("outburst.gui").joinpath("outburst.png")
        with as_file(resource) as path:
            image = QImage(str(path))
    except OSError:
        return QImage()
    return image


def build_app_icon() -> QIcon:
    """
    Load the Outburst icon for the window and the taskbar.

    Inputs: none.
    Output: a QIcon holding the artwork at 16, 24, 32, 48, 64, 128, and 256 pixels. An empty icon when the PNG cannot be read.
    Uses: the review window icon and the application icon set at startup.
    Limitations: smaller sizes are scaled from the 256-pixel file. A missing file leaves the window with no custom icon.
    """

    image = _icon_image()
    icon = QIcon()
    if image.isNull():
        return icon
    for size in (16, 24, 32, 48, 64, 128, 256):
        if image.width() == size and image.height() == size:
            icon.addPixmap(QPixmap.fromImage(image))
            continue
        scaled = image.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        icon.addPixmap(QPixmap.fromImage(scaled))
    return icon
