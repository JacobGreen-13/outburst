"""Large preview of the frame currently under review."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout


class PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._source = QPixmap()
        self._badge = ""
        self._badge_color = "#146b45"
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setWordWrap(True)

    def set_source(self, pixmap: QPixmap) -> None:
        self._source = pixmap
        if pixmap.isNull():
            self.setPixmap(QPixmap())
        else:
            self.setText("")
            self._rescale()

    def set_badge(self, text: str, color: str) -> None:
        self._badge = text
        self._badge_color = color

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source.isNull():
            return
        target = self.size()
        if target.width() < 2 or target.height() < 2:
            return
        scaled = self._source.scaled(
            target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self._badge:
            scaled = _draw_badge(scaled, self._badge, self._badge_color)
        self.setPixmap(scaled)


class ImageViewer(QFrame):
    keeper_clicked = Signal()
    reject_clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("viewerFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.preview = PreviewLabel()
        self.keeper_button = QPushButton("Mark as keeper")
        self.keeper_button.setObjectName("keeperButton")
        self.keeper_button.setCheckable(True)
        self.keeper_button.setToolTip(
            "Save this frame when you press Save keepers. Press again to clear it. Keyboard: A"
        )
        self.keeper_button.clicked.connect(lambda _checked=False: self.keeper_clicked.emit())
        self.reject_button = QPushButton("Reject")
        self.reject_button.setObjectName("rejectButton")
        self.reject_button.setCheckable(True)
        self.reject_button.setToolTip("Set this frame aside. It will not be saved. Keyboard: X")
        self.reject_button.clicked.connect(lambda _checked=False: self.reject_clicked.emit())
        marks = QHBoxLayout()
        marks.setSpacing(8)
        marks.addWidget(self.keeper_button, 1)
        marks.addWidget(self.reject_button, 1)
        self.info = QLabel()
        self.info.setObjectName("infoPanel")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setMinimumHeight(108)
        layout.addWidget(self.preview, 1)
        layout.addLayout(marks)
        layout.addWidget(self.info)
        self.set_placeholder(
            "Open a folder of RAW bursts to start culling.\n\n"
            "Outburst groups frames shot close together, scores sharpness and exposure "
            "from embedded previews, and marks the recommended frame as a keeper.\n\n"
            "Original files stay untouched."
        )

    def set_placeholder(self, text: str) -> None:
        self.preview.set_badge("", "#146b45")
        self.preview.set_source(QPixmap())
        self.preview.setText(text)
        self.info.setText("")
        self.set_border("#2a2a2a")
        self._apply_marks(False, False, enabled=False)

    def set_frame(
        self,
        preview: Path | None,
        box: tuple[int, int, int, int] | None,
        html: str,
        border: str,
        *,
        keeper: bool = False,
        rejected: bool = False,
    ) -> None:
        pixmap = QPixmap(str(preview)) if preview is not None else QPixmap()
        if rejected:
            self.preview.set_badge("REJECTED", "#8a3a3a")
        elif keeper:
            self.preview.set_badge("KEEPER", "#146b45")
        else:
            self.preview.set_badge("", "#146b45")
        if pixmap.isNull():
            self.preview.set_source(QPixmap())
            self.preview.setText("No preview for this frame")
        else:
            if box is not None:
                pixmap = _draw_box(pixmap, box)
            self.preview.set_source(pixmap)
        self.info.setText(html)
        self.set_border(border)
        self._apply_marks(keeper, rejected, enabled=True)

    def set_border(self, color: str) -> None:
        self.setStyleSheet(
            "QFrame#viewerFrame {"
            f"border: 2px solid {color}; border-radius: 10px; background: #141414;"
            "}"
            "QFrame#viewerFrame QLabel { color: #f2f2f2; background: transparent; }"
            "QLabel#infoPanel { background: #1b1b1b; }"
            "QPushButton#keeperButton, QPushButton#rejectButton {"
            "min-height: 36px; font-size: 14px; font-weight: 600; padding: 8px 16px;"
            "}"
            "QPushButton#keeperButton:checked {"
            "background: #1f6b45; color: #f4fff8; border: 1px solid #7dffc3;"
            "}"
            "QPushButton#rejectButton:checked {"
            "background: #6b2a2a; color: #fff5f5; border: 1px solid #e06c6c;"
            "}"
        )

    def _apply_marks(self, keeper: bool, rejected: bool, *, enabled: bool) -> None:
        self.keeper_button.blockSignals(True)
        self.reject_button.blockSignals(True)
        self.keeper_button.setChecked(keeper)
        self.reject_button.setChecked(rejected)
        self.keeper_button.setText("Marked keeper" if keeper else "Mark as keeper")
        self.reject_button.setText("Rejected" if rejected else "Reject")
        self.keeper_button.setEnabled(enabled)
        self.reject_button.setEnabled(enabled)
        self.keeper_button.blockSignals(False)
        self.reject_button.blockSignals(False)


def _draw_box(pixmap: QPixmap, box: tuple[int, int, int, int]) -> QPixmap:
    painted = pixmap.copy()
    painter = QPainter(painted)
    try:
        pen = QPen(QColor("#7dffc3"))
        pen.setWidth(max(2, painted.width() // 280))
        painter.setPen(pen)
        x, y, width, height = box
        painter.drawRect(x, y, width, height)
    finally:
        painter.end()
    return painted


def _draw_badge(pixmap: QPixmap, text: str, color: str) -> QPixmap:
    painted = pixmap.copy()
    painter = QPainter(painted)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(painter.font())
        font.setBold(True)
        font.setPixelSize(max(16, min(28, painted.width() // 28)))
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad_x = max(10, painted.width() // 70)
        pad_y = max(6, painted.height() // 90)
        margin = max(12, painted.width() // 48)
        rect = QRect(
            margin,
            margin,
            metrics.horizontalAdvance(text) + pad_x * 2,
            metrics.height() + pad_y * 2,
        )
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(QColor("#ffffff"))
        painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)
    finally:
        painter.end()
    return painted
