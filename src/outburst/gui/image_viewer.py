"""Large preview of the frame currently under review."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from outburst.scoring import focus_peak_bgr


class PreviewLabel(QLabel):
    clicked = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._source = QPixmap()
        self._peaked = QPixmap()
        self._badge = ""
        self._badge_color = "#146b45"
        self._box: tuple[int, int, int, int] | None = None
        self._circle: tuple[int, int, int] | None = None
        self._focus: tuple[int, int] | None = None
        self._identity = ""
        self._zoom = False
        self._peak = False
        self._origin_x = 0
        self._origin_y = 0
        self._pressed: QPoint | None = None
        self._moved = False
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(280, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setWordWrap(True)

    def set_source(
        self,
        pixmap: QPixmap,
        *,
        box: tuple[int, int, int, int] | None = None,
        circle: tuple[int, int, int] | None = None,
        focus: tuple[int, int] | None = None,
        identity: str = "",
    ) -> None:
        changed = identity != self._identity or (not identity and pixmap.cacheKey() != self._source.cacheKey())
        self._identity = identity
        self._source = pixmap
        self._box = box
        self._circle = circle
        self._focus = focus
        if changed:
            self._peaked = QPixmap()
        if pixmap.isNull():
            self.setPixmap(QPixmap())
            return
        self.setText("")
        if changed and self._zoom:
            self._center_origin()
        self._rescale()

    def set_badge(self, text: str, color: str) -> None:
        self._badge = text
        self._badge_color = color

    def set_view(self, *, zoom: bool, peak: bool) -> None:
        turned_on = zoom and not self._zoom
        self._zoom = zoom
        if not peak:
            self._peaked = QPixmap()
        self._peak = peak
        if turned_on:
            self._center_origin()
        self.setCursor(
            Qt.CursorShape.OpenHandCursor if zoom and not self._source.isNull() else Qt.CursorShape.ArrowCursor
        )
        self._rescale()

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        super().resizeEvent(event)
        self._rescale()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = QPoint(int(event.position().x()), int(event.position().y()))
            self._moved = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if self._pressed is not None and self._zoom and event.buttons() & Qt.MouseButton.LeftButton:
            delta_x = int(event.position().x()) - self._pressed.x()
            delta_y = int(event.position().y()) - self._pressed.y()
            if abs(delta_x) + abs(delta_y) > 3:
                self._moved = True
            self._origin_x -= delta_x
            self._origin_y -= delta_y
            self._pressed = QPoint(int(event.position().x()), int(event.position().y()))
            self._clamp_origin(self._display_source())
            self._rescale()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        if event.button() == Qt.MouseButton.LeftButton and self._pressed is not None and not self._moved:
            self.clicked.emit()
        self._pressed = None
        super().mouseReleaseEvent(event)

    def _center_origin(self) -> None:
        source = self._display_source()
        if source.isNull():
            return
        focus = self._focus
        if focus is None and self._box is not None:
            x, y, width, height = self._box
            focus = (x + width // 2, y + height // 2)
        if focus is None:
            focus = (source.width() // 2, source.height() // 2)
        self._origin_x = int(focus[0] - self.width() / 2)
        self._origin_y = int(focus[1] - self.height() / 2)
        self._clamp_origin(source)

    def _clamp_origin(self, source: QPixmap) -> None:
        if source.isNull():
            return
        self._origin_x = min(max(0, self._origin_x), max(0, source.width() - self.width()))
        self._origin_y = min(max(0, self._origin_y), max(0, source.height() - self.height()))

    def _display_source(self) -> QPixmap:
        if self._source.isNull() or not self._peak:
            return self._source
        if self._peaked.isNull():
            self._peaked = _peak_pixmap(self._source)
        return self._peaked

    def _rescale(self) -> None:
        source = self._display_source()
        if source.isNull():
            return
        if self._box is not None:
            source = _draw_box(source, self._box)
        if self._circle is not None:
            source = _draw_circle(source, self._circle)
        target = self.size()
        if target.width() < 2 or target.height() < 2:
            return
        if self._zoom:
            self._clamp_origin(source)
            cropped = source.copy(QRect(self._origin_x, self._origin_y, target.width(), target.height()))
            shown = cropped
        else:
            shown = source.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        if self._badge:
            shown = _draw_badge(shown, self._badge, self._badge_color)
        self.setPixmap(shown)


class ImageViewer(QFrame):
    keeper_clicked = Signal()
    reject_clicked = Signal()
    partner_clicked = Signal()
    star_clicked = Signal(int)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("viewerFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.preview = PreviewLabel()
        self.partner = PreviewLabel()
        self.partner.clicked.connect(self.partner_clicked.emit)
        self.partner_caption = QLabel("Best in the burst")
        self.partner_caption.setObjectName("partnerCaption")
        self.partner_box = QWidget()
        partner_layout = QVBoxLayout(self.partner_box)
        partner_layout.setContentsMargins(0, 0, 0, 0)
        partner_layout.setSpacing(4)
        partner_layout.addWidget(self.partner_caption)
        partner_layout.addWidget(self.partner, 1)
        self.partner_box.hide()
        panes = QHBoxLayout()
        panes.setSpacing(8)
        panes.addWidget(self.preview, 1)
        panes.addWidget(self.partner_box, 1)
        self.hint = QLabel()
        self.hint.setObjectName("viewerHint")
        self.hint.hide()
        self.star_buttons: list[QPushButton] = []
        for number in range(1, 6):
            button = QPushButton("★")
            button.setObjectName("starButton")
            button.setToolTip(f"Set {number} star{'s' if number != 1 else ''}. Press again to clear. Keyboard: {number}")
            button.clicked.connect(lambda _checked=False, value=number: self.star_clicked.emit(value))
            self.star_buttons.append(button)
        self.keeper_button = QPushButton("Mark as keeper")
        self.keeper_button.setObjectName("keeperButton")
        self.keeper_button.setCheckable(True)
        self.keeper_button.setToolTip(
            "Flag this frame as a pick. Save keepers copies picked frames, and the XMP sidecar "
            "records the pick. Press again to clear it. Keyboard: A"
        )
        self.keeper_button.clicked.connect(lambda _checked=False: self.keeper_clicked.emit())
        self.reject_button = QPushButton("Reject")
        self.reject_button.setObjectName("rejectButton")
        self.reject_button.setCheckable(True)
        self.reject_button.setToolTip(
            "Flag this frame as a reject. The XMP sidecar records the reject. Keyboard: X"
        )
        self.reject_button.clicked.connect(lambda _checked=False: self.reject_clicked.emit())
        marks = QHBoxLayout()
        marks.setSpacing(8)
        for button in self.star_buttons:
            marks.addWidget(button)
        marks.addWidget(self.keeper_button, 1)
        marks.addWidget(self.reject_button, 1)
        self.info = QLabel()
        self.info.setObjectName("infoPanel")
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setMinimumHeight(108)
        layout.addLayout(panes, 1)
        layout.addWidget(self.hint)
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
        self.partner_box.hide()
        self.hint.hide()
        self.info.setText("")
        self.set_border("#2a2a2a")
        self._apply_marks(False, False, None, enabled=False)

    def set_view_mode(self, *, zoom: bool, peak: bool) -> None:
        self.preview.set_view(zoom=zoom, peak=peak)
        self.partner.set_view(zoom=zoom, peak=peak)
        notes: list[str] = []
        if zoom:
            notes.append("1:1 preview · drag to look around")
        if peak:
            notes.append("Green marks the strongest edges")
        self.hint.setText("  ·  ".join(notes))
        self.hint.setVisible(bool(notes))

    def set_compare(self, visible: bool) -> None:
        self.partner_box.setVisible(visible)

    def set_partner(
        self,
        preview: Path | None,
        box: tuple[int, int, int, int] | None,
        caption: str,
        *,
        circle: tuple[int, int, int] | None = None,
        keeper: bool = False,
        rejected: bool = False,
    ) -> None:
        self.partner_caption.setText(caption)
        pixmap = QPixmap(str(preview)) if preview is not None else QPixmap()
        if rejected:
            self.partner.set_badge("REJECTED", "#8a3a3a")
        elif keeper:
            self.partner.set_badge("KEEPER", "#146b45")
        else:
            self.partner.set_badge("", "#146b45")
        focus = (circle[0], circle[1]) if circle is not None else _box_center(box)
        identity = "" if preview is None else str(preview)
        if pixmap.isNull():
            self.partner.set_source(QPixmap(), identity=identity)
            self.partner.setText("No preview for this frame")
        else:
            self.partner.set_source(pixmap, box=box, circle=circle, focus=focus, identity=identity)

    def set_partner_message(self, caption: str, message: str) -> None:
        self.partner_caption.setText(caption)
        self.partner.set_badge("", "#146b45")
        self.partner.set_source(QPixmap())
        self.partner.setText(message)

    def set_frame(
        self,
        preview: Path | None,
        box: tuple[int, int, int, int] | None,
        html: str,
        border: str,
        *,
        keeper: bool = False,
        rejected: bool = False,
        stars: int | None = None,
        circle: tuple[int, int, int] | None = None,
    ) -> None:
        pixmap = QPixmap(str(preview)) if preview is not None else QPixmap()
        if rejected:
            self.preview.set_badge("REJECTED", "#8a3a3a")
        elif keeper:
            self.preview.set_badge("KEEPER", "#146b45")
        else:
            self.preview.set_badge("", "#146b45")
        focus = (circle[0], circle[1]) if circle is not None else _box_center(box)
        identity = "" if preview is None else str(preview)
        if pixmap.isNull():
            self.preview.set_source(QPixmap(), identity=identity)
            self.preview.setText("No preview for this frame")
        else:
            self.preview.set_source(pixmap, box=box, circle=circle, focus=focus, identity=identity)
        self.info.setText(html)
        self.set_border(border)
        self._apply_marks(keeper, rejected, stars, enabled=True)

    def set_border(self, color: str) -> None:
        self.setStyleSheet(
            "QFrame#viewerFrame {"
            f"border: 2px solid {color}; border-radius: 10px; background: #141414;"
            "}"
            "QFrame#viewerFrame QLabel { color: #f2f2f2; background: transparent; }"
            "QLabel#infoPanel { background: #1b1b1b; }"
            "QLabel#partnerCaption, QLabel#viewerHint { color: #c8c8c8; }"
            "QPushButton#keeperButton, QPushButton#rejectButton {"
            "min-height: 36px; font-size: 14px; font-weight: 600; padding: 8px 16px;"
            "}"
            "QPushButton#starButton {"
            "min-height: 36px; min-width: 36px; max-width: 42px; font-size: 18px; color: #6a6a6a; padding: 0;"
            "}"
            "QPushButton#starButton[filled=\"true\"] { color: #e6c15a; border: 1px solid #e6c15a; }"
            "QPushButton#keeperButton:checked {"
            "background: #1f6b45; color: #f4fff8; border: 1px solid #7dffc3;"
            "}"
            "QPushButton#rejectButton:checked {"
            "background: #6b2a2a; color: #fff5f5; border: 1px solid #e06c6c;"
            "}"
        )

    def _apply_marks(self, keeper: bool, rejected: bool, stars: int | None, *, enabled: bool) -> None:
        self.keeper_button.blockSignals(True)
        self.reject_button.blockSignals(True)
        self.keeper_button.setChecked(keeper)
        self.reject_button.setChecked(rejected)
        self.keeper_button.setText("Marked keeper" if keeper else "Mark as keeper")
        self.reject_button.setText("Rejected" if rejected else "Reject")
        self.keeper_button.setEnabled(enabled)
        self.reject_button.setEnabled(enabled)
        filled = 0 if stars is None else stars
        for index, button in enumerate(self.star_buttons, start=1):
            button.setEnabled(enabled)
            button.setProperty("filled", "true" if index <= filled else "false")
            style = button.style()
            style.unpolish(button)
            style.polish(button)
        self.keeper_button.blockSignals(False)
        self.reject_button.blockSignals(False)


def _box_center(box: tuple[int, int, int, int] | None) -> tuple[int, int] | None:
    if box is None:
        return None
    x, y, width, height = box
    return (x + width // 2, y + height // 2)


def _draw_circle(pixmap: QPixmap, circle: tuple[int, int, int]) -> QPixmap:
    painted = pixmap.copy()
    painter = QPainter(painted)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#e6c15a"))
        pen.setWidth(max(2, painted.width() // 280))
        painter.setPen(pen)
        x, y, radius = circle
        painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
    finally:
        painter.end()
    return painted


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


def _peak_pixmap(pixmap: QPixmap) -> QPixmap:
    image = _pixmap_to_bgr(pixmap)
    if image is None:
        return pixmap
    return _bgr_to_pixmap(focus_peak_bgr(image))


def _pixmap_to_bgr(pixmap: QPixmap) -> np.ndarray | None:
    if pixmap.isNull():
        return None
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    width = image.width()
    height = image.height()
    if width < 1 or height < 1:
        return None
    stride = image.bytesPerLine()
    raw = np.frombuffer(image.constBits(), dtype=np.uint8, count=stride * height)
    rgb = np.ascontiguousarray(raw.reshape((height, stride))[:, : width * 3].reshape((height, width, 3)))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())
