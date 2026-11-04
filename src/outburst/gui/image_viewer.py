"""Large preview of the frame currently under review."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QLabel, QSizePolicy, QVBoxLayout


class PreviewLabel(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self._source = QPixmap()
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
        self.setPixmap(scaled)


class ImageViewer(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("viewerFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 0)
        layout.setSpacing(0)
        self.preview = PreviewLabel()
        self.info = QLabel()
        self.info.setObjectName("infoPanel")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setMinimumHeight(108)
        layout.addWidget(self.preview, 1)
        layout.addWidget(self.info)
        self.set_placeholder(
            "Open a folder of RAW bursts to start culling.\n\n"
            "Outburst groups frames shot close together, scores sharpness and exposure "
            "from embedded previews, and highlights a recommended frame.\n\n"
            "Original files stay untouched."
        )

    def set_placeholder(self, text: str) -> None:
        self.preview.set_source(QPixmap())
        self.preview.setText(text)
        self.info.setText("")
        self.set_border("#2a2a2a")

    def set_frame(
        self,
        preview: Path | None,
        box: tuple[int, int, int, int] | None,
        html: str,
        border: str,
    ) -> None:
        pixmap = QPixmap(str(preview)) if preview is not None else QPixmap()
        if pixmap.isNull():
            self.preview.set_source(QPixmap())
            self.preview.setText("No preview for this frame")
        else:
            if box is not None:
                pixmap = _draw_box(pixmap, box)
            self.preview.set_source(pixmap)
        self.info.setText(html)
        self.set_border(border)

    def set_border(self, color: str) -> None:
        self.setStyleSheet(
            "QFrame#viewerFrame {"
            f"border: 2px solid {color}; border-radius: 10px; background: #141414;"
            "}"
            "QFrame#viewerFrame QLabel { color: #f2f2f2; background: transparent; }"
            "QLabel#infoPanel { background: #1b1b1b; }"
        )


def _draw_box(pixmap: QPixmap, box: tuple[int, int, int, int]) -> QPixmap:
    painted = pixmap.copy()
    painter = QPainter(painted)
    pen = QPen(QColor("#7dffc3"))
    pen.setWidth(max(2, painted.width() // 280))
    painter.setPen(pen)
    x, y, width, height = box
    painter.drawRect(x, y, width, height)
    painter.end()
    return painted
