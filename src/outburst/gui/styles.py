"""Dark review theme. A dim surround makes it easier to judge a photograph."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
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
QLabel#subtitle, QLabel#folderLabel, QLabel#hint {
    color: #a3a3a3;
    background: transparent;
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
    padding: 10px 8px 8px 8px;
    background: #1c1c1c;
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
QDoubleSpinBox, QLineEdit, QComboBox {
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
QSplitter::handle {
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


def build_app_icon() -> QIcon:
    """A small stacked-frame mark, drawn in code so the repo stays free of binary assets."""

    size = 256
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#143026"))
    painter.drawRoundedRect(0, 0, size, size, 56, 56)
    pen = QPen(QColor("#d7efe4"))
    pen.setWidth(8)
    painter.setPen(pen)
    painter.setBrush(QColor("#1f6b4a"))
    for rect in ((96, 40, 118, 92), (72, 62, 118, 92), (48, 84, 118, 92)):
        painter.drawRoundedRect(*rect, 12, 12)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#f2c14e"))
    painter.drawEllipse(112, 118, 30, 22)
    painter.end()
    return QIcon(pixmap)
