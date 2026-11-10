"""Large preview of the frame currently under review.

``next_zoom`` steps through fit, 1:1, 2:1, and 4:1. ``PreviewLabel`` pans and crops the embedded preview. ``ImageViewer.set_frame`` shows one frame, ``set_partner`` the compare frame, and ``set_view_mode`` applies zoom and focus peaking to both. The Info button shows the measurements after the Laplacian number and starts off. ``_draw_box``, ``_draw_circle``, and ``_draw_badge`` mark the subject, the eye, and the keeper on the preview. Zoom does not demosaic the RAW file.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from outburst.scoring import focus_peak_bgr


_ZOOM_STEPS = (1.0, 2.0, 4.0)


def next_zoom(scale: float | None, steps: int) -> float | None:
    """
    Move `steps` along fit, 1:1, 2:1, and 4:1. A negative step from 1:1 returns to fit.

    Inputs: scale, the current zoom, or None for fit; steps, how many stops to move. A negative steps value zooms out.
    Output: the next scale of 1, 2, or 4, or None for fit. A zero step returns scale unchanged.
    Uses: preview zoom from the keyboard and the scroll wheel.
    Limitations: zoom steps are fit, 1, 2, and 4 and use the embedded preview. A scale that is not on a step snaps to the nearest step before moving.
    """

    if steps == 0:
        return scale
    if scale is None:
        return _ZOOM_STEPS[0] if steps > 0 else None
    nearest = min(range(len(_ZOOM_STEPS)), key=lambda index: abs(_ZOOM_STEPS[index] - scale))
    if steps < 0 and nearest == 0:
        return None
    chosen = max(0, min(len(_ZOOM_STEPS) - 1, nearest + steps))
    return _ZOOM_STEPS[chosen]


class PreviewLabel(QLabel):
    clicked = Signal()
    zoom_stepped = Signal(int, QPoint)

    def __init__(self) -> None:
        """
        Build an empty preview that can fit, zoom, and pan.

        Inputs: none.
        Output: no return value. The label is empty, at fit scale, with peaking off and no badge.
        Uses: the main preview and the compare preview.
        Limitations: the minimum size is 280 by 220. There is no image until set_source is called.
        """
        super().__init__()
        self._source = QPixmap()
        self._peaked = QPixmap()
        self._badge = ""
        self._badge_color = "#146b45"
        self._box: tuple[int, int, int, int] | None = None
        self._circle: tuple[int, int, int] | None = None
        self._focus: tuple[int, int] | None = None
        self._identity = ""
        self._scale: float | None = None
        self._peak = False
        self._origin_x = 0.0
        self._origin_y = 0.0
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
        """
        Replace the preview image and the subject box, eye circle, and focus point.

        Inputs: pixmap, the embedded preview; box, a subject rectangle or None; circle, an eye circle or None; focus, a point to center when zoomed, or None; identity, a path string used to tell frames apart.
        Output: no return value. A null pixmap clears the image. A changed frame drops the cached peak overlay and recenters when already zoomed.
        Uses: the large preview of the current frame and the compare frame.
        Limitations: zoom uses the embedded preview. An empty identity treats a different pixmap cache key as a new frame.
        """
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
        if changed and self._scale is not None:
            self._center_origin()
        self._rescale()

    def scale(self) -> float | None:
        """
        Read the current zoom scale.

        Inputs: none.
        Output: 1, 2, or 4 when zoomed, or None when the frame is fit to the label.
        Uses: zoom stepping and the preview hint.
        Limitations: None means fit. It is not a numeric scale of 1.
        """
        return self._scale

    def peak_enabled(self) -> bool:
        """
        Read whether focus peaking is on for this label.

        Inputs: none.
        Output: True when green edge marks are requested.
        Uses: the preview hint and keeping both panes in the same peaking state.
        Limitations: the green overlay is built the next time the preview is drawn, not in this method.
        """
        return self._peak

    def set_badge(self, text: str, color: str) -> None:
        """
        Store the keeper or rejected badge drawn on the preview.

        Inputs: text, the badge words, or empty to hide it; color, the badge background.
        Output: no return value. The badge text and color are stored for the next rescale.
        Uses: the KEEPER and REJECTED marks on the preview.
        Limitations: this method does not redraw. The badge appears when the preview is scaled again.
        """
        self._badge = text
        self._badge_color = color

    def set_peak(self, peak: bool) -> None:
        """
        Turn focus peaking on or off and redraw.

        Inputs: peak, whether to paint the strongest edges green.
        Output: no return value. Turning peaking off drops the cached overlay. A changed value redraws the preview.
        Uses: focus peaking on one preview pane.
        Limitations: peaking uses the embedded preview. An unchanged value returns without redrawing.
        """
        if not peak:
            self._peaked = QPixmap()
        if peak == self._peak:
            return
        self._peak = peak
        self._rescale()

    def set_scale(self, scale: float | None, *, anchor: QPoint | None = None) -> None:
        """
        Set fit or a zoom scale, keeping an anchor point stable when one is given.

        Inputs: scale, 1, 2, or 4, or None for fit; anchor, a point in this label to keep under the cursor, or None.
        Output: no return value. The pan origin is adjusted, the cursor updates, and the preview is redrawn.
        Uses: zoom and pan of the embedded preview.
        Limitations: the same scale with no anchor returns without redrawing. A null source forces fit. Entering zoom with no anchor centers on the focus point, subject, or image center.
        """
        if scale == self._scale and anchor is None:
            return
        source = self._display_source()
        if scale is None or source.isNull():
            self._scale = None
            self._apply_cursor()
            self._rescale()
            return
        entering = self._scale is None
        if entering and anchor is not None:
            fit = self._fit_scale(source)
            shown_width = source.width() * fit
            shown_height = source.height() * fit
            left = (self.width() - shown_width) / 2
            top = (self.height() - shown_height) / 2
            source_x = (anchor.x() - left) / fit if fit else source.width() / 2
            source_y = (anchor.y() - top) / fit if fit else source.height() / 2
            self._scale = scale
            self._origin_x = source_x - anchor.x() / scale
            self._origin_y = source_y - anchor.y() / scale
        elif entering:
            self._scale = scale
            self._center_origin()
        elif anchor is not None and self._scale and self._scale > 0:
            source_x = self._origin_x + anchor.x() / self._scale
            source_y = self._origin_y + anchor.y() / self._scale
            self._scale = scale
            self._origin_x = source_x - anchor.x() / scale
            self._origin_y = source_y - anchor.y() / scale
        elif self._scale and self.width() > 1:
            center_x = self._origin_x + (self.width() / 2) / self._scale
            center_y = self._origin_y + (self.height() / 2) / self._scale
            self._scale = scale
            self._origin_x = center_x - (self.width() / 2) / scale
            self._origin_y = center_y - (self.height() / 2) / scale
        else:
            self._scale = scale
        self._apply_cursor()
        self._rescale()

    def _apply_cursor(self) -> None:
        """
        Show an open hand while a zoomed image can be dragged.

        Inputs: none.
        Output: no return value. The cursor is an open hand when zoomed onto a real image, and an arrow otherwise.
        Uses: the preview cursor.
        Limitations: fit mode uses the arrow cursor, including when a pixmap is present.
        """
        zoomed = self._scale is not None and not self._source.isNull()
        self.setCursor(Qt.CursorShape.OpenHandCursor if zoomed else Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Redraw the preview after the label changes size.

        Inputs: event, the resize event.
        Output: no return value. The base handler runs, then the current scale is drawn again.
        Uses: the preview layout.
        Limitations: the zoom scale is kept. The crop origin stays where it was and is clamped to the new label size.
        """
        super().resizeEvent(event)
        self._rescale()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Remember a left-button press so a drag can pan.

        Inputs: event, the mouse press.
        Output: no return value. A left press stores the position and clears the moved flag, then the base handler runs.
        Uses: panning a zoomed preview.
        Limitations: only the left button is tracked. The press does not emit clicked.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = QPoint(int(event.position().x()), int(event.position().y()))
            self._moved = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Pan a zoomed preview while the left button is dragged.

        Inputs: event, the mouse move.
        Output: no return value. When a press is active and the view is zoomed, the origin follows the drag and the preview is redrawn.
        Uses: looking around a zoomed preview.
        Limitations: panning runs only while zoomed and the left button is down. A combined horizontal and vertical movement of more than 3 pixels marks the gesture as a drag.
        """
        if self._pressed is not None and self._scale and event.buttons() & Qt.MouseButton.LeftButton:
            delta_x = int(event.position().x()) - self._pressed.x()
            delta_y = int(event.position().y()) - self._pressed.y()
            if abs(delta_x) + abs(delta_y) > 3:
                self._moved = True
            self._origin_x -= delta_x / self._scale
            self._origin_y -= delta_y / self._scale
            self._pressed = QPoint(int(event.position().x()), int(event.position().y()))
            self._clamp_origin(self._display_source())
            self._rescale()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Emit a click when the left button is released without a drag.

        Inputs: event, the mouse release.
        Output: no return value. A left release that was not a drag emits clicked. The stored press is cleared.
        Uses: a click on the compare preview, which the viewer connects to swapping sides.
        Limitations: a drag of more than 3 pixels does not emit clicked. The main preview's click signal is not connected by the viewer.
        """
        if event.button() == Qt.MouseButton.LeftButton and self._pressed is not None and not self._moved:
            self.clicked.emit()
        self._pressed = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Ask the viewer to zoom one step toward the cursor.

        Inputs: event, the wheel event.
        Output: no return value. A vertical wheel delta emits zoom_stepped with +1 or -1 and the cursor point, and the event is accepted.
        Uses: zooming the embedded preview.
        Limitations: zoom steps are fit, 1, 2, and 4. A zero vertical delta, or an empty preview, is left to the base handler. Horizontal wheel movement is ignored.
        """
        delta = event.angleDelta().y()
        if delta == 0 or self._source.isNull():
            super().wheelEvent(event)
            return
        step = 1 if delta > 0 else -1
        anchor = QPoint(int(event.position().x()), int(event.position().y()))
        self.zoom_stepped.emit(step, anchor)
        event.accept()

    def _center_origin(self) -> None:
        """
        Place the zoom origin so the focus point sits in the middle of the label.

        Inputs: none. The stored focus is used, then the subject-box center, then the image center.
        Output: no return value. The origin is set and clamped inside the image.
        Uses: the first zoom into a frame, and a new frame opened while already zoomed.
        Limitations: an empty source returns immediately. A missing scale is treated as 1.
        """
        source = self._display_source()
        if source.isNull():
            return
        focus = self._focus
        if focus is None and self._box is not None:
            x, y, width, height = self._box
            focus = (x + width // 2, y + height // 2)
        if focus is None:
            focus = (source.width() // 2, source.height() // 2)
        scale = self._scale or 1.0
        self._origin_x = focus[0] - (self.width() / 2) / scale
        self._origin_y = focus[1] - (self.height() / 2) / scale
        self._clamp_origin(source)

    def _fit_scale(self, source: QPixmap) -> float:
        """
        Compute the scale that fits the source inside the label.

        Inputs: source, the pixmap being fitted.
        Output: the smaller of the width ratio and the height ratio, or 1.0 when the source or the label is smaller than 1 pixel.
        Uses: zooming in from fit while keeping the cursor point.
        Limitations: the returned scale is the fit ratio. It is not one of the 1, 2, or 4 zoom steps.
        """
        if source.width() < 1 or source.height() < 1 or self.width() < 1 or self.height() < 1:
            return 1.0
        return min(self.width() / source.width(), self.height() / source.height())

    def _clamp_origin(self, source: QPixmap, crop_width: int | None = None, crop_height: int | None = None) -> None:
        """
        Keep the zoom crop inside the source image.

        Inputs: source, the pixmap; crop_width and crop_height, the crop in source pixels, or None to derive them from the label and the current scale.
        Output: no return value. The origin is clamped so the crop stays inside the image.
        Uses: pan and zoom of the preview.
        Limitations: a null source is ignored. A missing scale is treated as 1.
        """
        if source.isNull():
            return
        scale = self._scale or 1.0
        width = crop_width if crop_width is not None else max(1, int(round(self.width() / scale)))
        height = crop_height if crop_height is not None else max(1, int(round(self.height() / scale)))
        self._origin_x = min(max(0.0, self._origin_x), max(0.0, float(source.width() - width)))
        self._origin_y = min(max(0.0, self._origin_y), max(0.0, float(source.height() - height)))

    def _display_source(self) -> QPixmap:
        """
        Return the preview pixmap, with focus peaking applied when it is on.

        Inputs: none.
        Output: the original pixmap, or a cached peaked pixmap when peaking is on and a source exists.
        Uses: drawing the preview.
        Limitations: the peaked image is built once per source and reused until the source changes. Peaking uses the embedded preview.
        """
        if self._source.isNull() or not self._peak:
            return self._source
        if self._peaked.isNull():
            self._peaked = _peak_pixmap(self._source)
        return self._peaked

    def _rescale(self) -> None:
        """
        Draw the current crop, overlays, and badge into the label.

        Inputs: none.
        Output: no return value. The label pixmap is replaced. Fit scales the whole frame. Zoom copies a crop and scales that crop.
        Uses: the large preview.
        Limitations: zoom uses the embedded preview. A label smaller than 2 pixels is skipped. The subject box and eye circle are drawn on the source before the crop. The badge is drawn on the scaled image.
        """
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
        if self._scale is not None:
            crop_width = max(1, int(round(target.width() / self._scale)))
            crop_height = max(1, int(round(target.height() / self._scale)))
            self._clamp_origin(source, crop_width, crop_height)
            cropped = source.copy(QRect(int(self._origin_x), int(self._origin_y), crop_width, crop_height))
            shown = cropped.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
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
    zoom_changed = Signal(float)

    def __init__(self) -> None:
        """
        Build the preview, compare pane, star buttons, and keeper and reject buttons.

        Inputs: none.
        Output: no return value. The compare pane starts hidden, Info starts off, and the welcome placeholder is shown.
        Uses: the main review pane.
        Limitations: the five star buttons emit 1 through 5. Pressing the same rating again is cleared by the window. The compare pane's click is connected; the main preview's click is not. Info reveals the detail label and does not rescore.
        """
        super().__init__()
        self.setObjectName("viewerFrame")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.preview = PreviewLabel()
        self.partner = PreviewLabel()
        self.partner.clicked.connect(self.partner_clicked.emit)
        self.preview.zoom_stepped.connect(lambda step, anchor: self._zoom_from(self.preview, step, anchor))
        self.partner.zoom_stepped.connect(lambda step, anchor: self._zoom_from(self.partner, step, anchor))
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
        self.keeper_button = QPushButton("Keep · A")
        self.keeper_button.setObjectName("keeperButton")
        self.keeper_button.setCheckable(True)
        self.keeper_button.setToolTip(
            "Flag this frame as a pick. Save keepers copies picked frames and writes an .xmp "
            "sidecar beside each one. Press again to clear it. Keyboard: A"
        )
        self.keeper_button.clicked.connect(lambda _checked=False: self.keeper_clicked.emit())
        self.reject_button = QPushButton("Reject · X")
        self.reject_button.setObjectName("rejectButton")
        self.reject_button.setCheckable(True)
        self.reject_button.setToolTip(
            "Flag this frame as a reject. The photograph stays where it is, and Save keepers skips it. "
            "A star rating on this frame is still written to a sidecar. Keyboard: X"
        )
        self.reject_button.clicked.connect(lambda _checked=False: self.reject_clicked.emit())
        marks = QHBoxLayout()
        marks.setSpacing(6)
        rating = QLabel("Rating")
        rating.setObjectName("fieldLabel")
        marks.addWidget(rating)
        for button in self.star_buttons:
            marks.addWidget(button)
        marks.addStretch(1)
        marks.addWidget(self.keeper_button)
        marks.addWidget(self.reject_button)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextFormat(Qt.TextFormat.RichText)
        self.info.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.detail = QLabel()
        self.detail.setObjectName("infoDetail")
        self.detail.setWordWrap(True)
        self.detail.setTextFormat(Qt.TextFormat.RichText)
        self.detail.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.detail.hide()
        self.info_button = QPushButton("Info")
        self.info_button.setObjectName("infoButton")
        self.info_button.setCheckable(True)
        self.info_button.setChecked(False)
        self.info_button.setToolTip(
            "Show the frame, exposure, and measurements after the Laplacian number."
        )
        self.info_button.toggled.connect(self._on_info_toggled)
        self.info_box = QWidget()
        self.info_box.setObjectName("infoPanel")
        info_text = QVBoxLayout(self.info_box)
        info_text.setContentsMargins(8, 4, 8, 4)
        info_text.setSpacing(0)
        info_text.addWidget(self.info)
        info_text.addWidget(self.detail)
        info_row = QHBoxLayout()
        info_row.setSpacing(8)
        info_row.addWidget(self.info_box, 1)
        info_row.addWidget(self.info_button, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(panes, 1)
        layout.addWidget(self.hint)
        layout.addLayout(marks)
        layout.addLayout(info_row)
        self.set_placeholder(
            "Open a folder of RAW bursts to start culling.\n\n"
            "Outburst groups frames shot close together, scores sharpness and exposure "
            "from embedded previews, and marks the recommended frame as a keeper.\n\n"
            "Original files stay untouched."
        )

    def set_placeholder(self, text: str) -> None:
        """
        Replace the preview with a message and clear the marks.

        Inputs: text, the message shown in the preview.
        Output: no return value. The badge, image, partner pane, hint, and info are cleared, Info is hidden, and the mark buttons are disabled.
        Uses: the empty review window and a folder whose bursts are all shorter than Minimum frames.
        Limitations: the border returns to the idle color. Keeper, reject, and star controls do not accept clicks while the placeholder is shown.
        """
        self.preview.set_badge("", "#146b45")
        self.preview.set_source(QPixmap())
        self.preview.setText(text)
        self.partner_box.hide()
        self.hint.hide()
        self.info.setText("")
        self.detail.setText("")
        self.detail.hide()
        self.info_box.hide()
        self.info_button.hide()
        self.set_border("#2a2a2a")
        self._apply_marks(False, False, None, enabled=False)

    def set_view_mode(self, *, scale: float | None, peak: bool) -> None:
        """
        Apply one zoom scale and peaking state to both previews.

        Inputs: scale, 1, 2, or 4, or None for fit; peak, whether focus peaking is on.
        Output: no return value. Both panes update and the hint lists the mode.
        Uses: Zoom and Focus peaking for the current frame and the compare frame.
        Limitations: both panes share the same scale. Neither pane is given a cursor anchor, so a newly entered zoom centers on the focus point.
        """
        self.preview.set_peak(peak)
        self.partner.set_peak(peak)
        self.preview.set_scale(scale)
        self.partner.set_scale(scale)
        self._show_hint(scale, peak)

    def _zoom_from(self, source: PreviewLabel, step: int, anchor: QPoint) -> None:
        """
        Step the shared zoom from the pane under the cursor and keep that cursor point.

        Inputs: source, the preview that received the wheel; step, +1 or -1; anchor, the cursor point in that pane.
        Output: no return value. Both panes take the next scale, the hint updates, and zoom_changed emits 0 for fit or the new scale.
        Uses: the scroll wheel over either preview.
        Limitations: zoom steps are fit, 1, 2, and 4 and use the embedded preview. Only the source pane keeps the anchor. The other pane is updated without an anchor.
        """
        scale = next_zoom(self.preview.scale(), step)
        if source is self.preview:
            self.preview.set_scale(scale, anchor=anchor)
            self.partner.set_scale(scale)
        else:
            self.partner.set_scale(scale, anchor=anchor)
            self.preview.set_scale(scale)
        self._show_hint(scale, self.preview.peak_enabled())
        self.zoom_changed.emit(0.0 if scale is None else scale)

    def _show_hint(self, scale: float | None, peak: bool) -> None:
        """
        Show the zoom and focus-peaking note under the preview.

        Inputs: scale, the current zoom or None for fit; peak, whether peaking is on.
        Output: no return value. The hint lists the notes that apply, and it is hidden when the list is empty.
        Uses: the line under the preview.
        Limitations: fit with peaking off hides the hint. A scale of 1 is labeled 1:1. Any other numeric scale is labeled with its integer ratio.
        """
        notes: list[str] = []
        if scale == 1:
            notes.append("1:1 preview · drag to look around")
        elif scale is not None:
            notes.append(f"{int(scale)}:1 preview · drag to look around")
        if peak:
            notes.append("Green marks the strongest edges")
        self.hint.setText("  ·  ".join(notes))
        self.hint.setVisible(bool(notes))

    def set_compare(self, visible: bool) -> None:
        """
        Show or hide the second preview.

        Inputs: visible, whether compare is on.
        Output: no return value. The partner pane is shown or hidden.
        Uses: Compare in the review window.
        Limitations: hiding the pane does not clear the partner image or caption.
        """
        self.partner_box.setVisible(visible)

    def set_partner(
        self,
        preview: Path | None,
        box: tuple[int, int, int, int] | None,
        caption: str,
        *,
        circle: tuple[int, int, int] | None = None,
        focus: tuple[int, int] | None = None,
        keeper: bool = False,
        rejected: bool = False,
    ) -> None:
        """
        Show the other frame in the compare pane.

        Inputs: preview, the embedded preview path or None; box, the subject rectangle or None; caption, the line above the pane; circle, the eye circle or None; focus, a zoom center, or None to use the circle then the box; keeper and rejected, which badge to draw.
        Output: no return value. The caption, badge, and image update. A missing or unreadable preview shows "No preview for this frame".
        Uses: the compare pane.
        Limitations: zoom uses the embedded preview. A rejected badge is drawn ahead of a keeper badge. A missing focus centers on the eye circle when one is given, otherwise on the subject box. A box or circle that is omitted is not drawn.
        """
        self.partner_caption.setText(caption)
        pixmap = QPixmap(str(preview)) if preview is not None else QPixmap()
        if rejected:
            self.partner.set_badge("REJECTED", "#8a3a3a")
        elif keeper:
            self.partner.set_badge("KEEPER", "#146b45")
        else:
            self.partner.set_badge("", "#146b45")
        if focus is None:
            focus = (circle[0], circle[1]) if circle is not None else _box_center(box)
        identity = "" if preview is None else str(preview)
        if pixmap.isNull():
            self.partner.set_source(QPixmap(), identity=identity)
            self.partner.setText("No preview for this frame")
        else:
            self.partner.set_source(pixmap, box=box, circle=circle, focus=focus, identity=identity)

    def set_partner_message(self, caption: str, message: str) -> None:
        """
        Replace the compare pane with a caption and a text message.

        Inputs: caption, the line above the pane; message, the text in the pane.
        Output: no return value. The badge and image are cleared and the message is shown.
        Uses: compare when a burst has one frame.
        Limitations: this does not hide the pane. The caller decides whether compare is visible.
        """
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
        focus: tuple[int, int] | None = None,
        detail: str = "",
    ) -> None:
        """
        Show one frame in the main preview with its info and marks.

        Inputs: preview, the embedded preview path or None; box, the subject rectangle or None; html, the summary through the Laplacian number; border, the frame color; keeper, rejected, and stars, the marks; circle, the eye circle or None; focus, a zoom center, or None to use the circle then the box; detail, the measurements after the Laplacian number.
        Output: no return value. The preview, summary, detail, border, and buttons update. A missing or unreadable preview shows "No preview for this frame". The mark buttons are enabled.
        Uses: the frame currently under review.
        Limitations: zoom uses the embedded preview. A rejected badge is drawn ahead of a keeper badge. A missing focus centers on the eye circle when one is given, otherwise on the subject box. A box or circle that is omitted is not drawn. The detail stays hidden until Info is on.
        """
        pixmap = QPixmap(str(preview)) if preview is not None else QPixmap()
        if rejected:
            self.preview.set_badge("REJECTED", "#8a3a3a")
        elif keeper:
            self.preview.set_badge("KEEPER", "#146b45")
        else:
            self.preview.set_badge("", "#146b45")
        if focus is None:
            focus = (circle[0], circle[1]) if circle is not None else _box_center(box)
        identity = "" if preview is None else str(preview)
        if pixmap.isNull():
            self.preview.set_source(QPixmap(), identity=identity)
            self.preview.setText("No preview for this frame")
        else:
            self.preview.set_source(pixmap, box=box, circle=circle, focus=focus, identity=identity)
        self.info.setText(html)
        self.detail.setText(detail)
        self.info_box.show()
        self.info_button.show()
        self._apply_info_visibility()
        self.set_border(border)
        self._apply_marks(keeper, rejected, stars, enabled=True)

    def info_open(self) -> bool:
        """
        Whether the measurements after the Laplacian number are shown.

        Inputs: none.
        Output: True when Info is checked.
        Uses: saving the review window settings.
        Limitations: this is the button state. A placeholder hides the button and leaves this value as it was.
        """
        return self.info_button.isChecked()

    def set_info_open(self, open_: bool) -> None:
        """
        Show or hide the measurements after the Laplacian number.

        Inputs: open_, whether those lines are shown.
        Output: no return value. The button matches open_ and the detail label follows it.
        Uses: restoring the Info button from settings.
        Limitations: the signal is blocked, so restoring does not redraw. An empty detail stays hidden.
        """
        self.info_button.blockSignals(True)
        self.info_button.setChecked(bool(open_))
        self.info_button.blockSignals(False)
        self._apply_info_visibility()

    def _on_info_toggled(self, _checked: bool) -> None:
        """
        Show or hide the detail when Info is clicked.

        Inputs: _checked, the new button state. The button is read again while the label is updated.
        Output: no return value. The detail label is shown only when Info is on and it has text.
        Uses: the Info button beside the score.
        Limitations: this does not rescore and does not change the summary.
        """
        self._apply_info_visibility()

    def _apply_info_visibility(self) -> None:
        """
        Match the detail label to the Info button.

        Inputs: none.
        Output: no return value. The detail is visible when Info is checked and the label has text.
        Uses: set_frame, set_info_open, and the Info button.
        Limitations: a placeholder that has hidden the button leaves the detail hidden.
        """
        self.detail.setVisible(self.info_button.isChecked() and bool(self.detail.text()))

    def set_border(self, color: str) -> None:
        """
        Apply the preview frame color and the styles for the info panel and mark buttons.

        Inputs: color, the border color.
        Output: no return value. The viewer stylesheet is replaced.
        Uses: the review pane border for rejected, keeper, recommended, and other frames.
        Limitations: the call replaces the whole viewer stylesheet, including the mark-button rules stored in this method.
        """
        self.setStyleSheet(
            "QFrame#viewerFrame {"
            f"border: 2px solid {color}; border-radius: 10px; background: #141414;"
            "}"
            "QFrame#viewerFrame QLabel { color: #f2f2f2; background: transparent; }"
            "QWidget#infoPanel { background: #1b1b1b; border-radius: 6px; }"
            "QLabel#infoDetail { background: transparent; }"
            "QPushButton#infoButton {"
            "min-height: 22px; max-height: 28px; padding: 2px 10px; font-size: 12px; font-weight: 600;"
            "}"
            "QPushButton#infoButton:checked {"
            "background: #1f3d30; color: #f4fff8; border: 1px solid #7dffc3;"
            "}"
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
        """
        Match the keeper, reject, and star buttons to the current frame.

        Inputs: keeper and rejected, the checked states; stars, 1 through 5 or None; enabled, whether the buttons accept clicks.
        Output: no return value. Labels, checks, and filled stars update. Keeper and reject signals are blocked during the update.
        Uses: the marks under the preview.
        Limitations: a missing star rating fills no stars. Choosing the same rating again is handled by the window, which clears it, before this method is called.
        """
        self.keeper_button.blockSignals(True)
        self.reject_button.blockSignals(True)
        self.keeper_button.setChecked(keeper)
        self.reject_button.setChecked(rejected)
        self.keeper_button.setText("Keep · A")
        self.reject_button.setText("Reject · X")
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
    """
    Return the center of a subject box.

    Inputs: box, x, y, width, and height, or None.
    Output: the center point, or None when box is None.
    Uses: choosing where a zoomed preview centers when no eye circle is present.
    Limitations: the center uses integer division of the width and height.
    """
    if box is None:
        return None
    x, y, width, height = box
    return (x + width // 2, y + height // 2)


def _draw_circle(pixmap: QPixmap, circle: tuple[int, int, int]) -> QPixmap:
    """
    Draw the eye circle onto a copy of the preview.

    Inputs: pixmap, the source; circle, center x, center y, and radius.
    Output: a new pixmap with a gold ring and the word EYE. The painter is closed before the pixmap is returned.
    Uses: the eye mark on the preview.
    Limitations: the source pixmap is copied and left unchanged. The pen width and the label size grow with the preview. The label moves to the left of the ring when it would leave the frame.
    """
    painted = pixmap.copy()
    painter = QPainter(painted)
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        x, y, radius = circle
        radius = max(1, int(radius))
        halo = QPen(QColor("#1a1206"))
        halo.setWidth(max(4, painted.width() // 140))
        painter.setPen(halo)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        pen = QPen(QColor("#ffe08a"))
        pen.setWidth(max(2, painted.width() // 240))
        painter.setPen(pen)
        painter.drawEllipse(x - radius, y - radius, radius * 2, radius * 2)
        font = QFont()
        font.setPixelSize(max(16, radius))
        font.setBold(True)
        painter.setFont(font)
        label = "EYE"
        label_width = max(radius * 2, 36)
        label_x = x + radius + max(8, radius // 4)
        if label_x + label_width > painted.width():
            label_x = max(4, x - radius - label_width)
        label_y = y + font.pixelSize() // 3
        painter.setPen(QColor("#1a1206"))
        for offset_x, offset_y in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            painter.drawText(label_x + offset_x, label_y + offset_y, label)
        painter.setPen(QColor("#ffe08a"))
        painter.drawText(label_x, label_y, label)
    finally:
        painter.end()
    return painted


def _draw_box(pixmap: QPixmap, box: tuple[int, int, int, int]) -> QPixmap:
    """
    Draw the subject rectangle onto a copy of the preview.

    Inputs: pixmap, the source; box, x, y, width, and height.
    Output: a new pixmap with a green rectangle. The painter is closed before the pixmap is returned.
    Uses: the subject mark on the preview.
    Limitations: the source pixmap is copied and left unchanged. The pen width grows with the image width, with a minimum of 2.
    """
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
    """
    Draw a badge in the corner of a copy of the preview.

    Inputs: pixmap, the shown image; text, the badge words; color, the badge background.
    Output: a new pixmap with a rounded rectangle and white centered text.
    Uses: the KEEPER and REJECTED badge on the preview.
    Limitations: the badge is drawn on the already scaled image, so its pixel size follows the label rather than the zoom crop.
    """
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
    """
    Paint the strongest edges of a preview in green.

    Inputs: pixmap, the embedded preview.
    Output: a peaked pixmap, or the original pixmap when it cannot be converted to an image.
    Uses: focus peaking.
    Limitations: peaking uses the embedded preview. A pixmap that cannot be read as RGB is returned unchanged.
    """
    image = _pixmap_to_bgr(pixmap)
    if image is None:
        return pixmap
    return _bgr_to_pixmap(focus_peak_bgr(image))


def _pixmap_to_bgr(pixmap: QPixmap) -> np.ndarray | None:
    """
    Convert a pixmap to a BGR image for focus peaking.

    Inputs: pixmap, the preview.
    Output: a contiguous BGR array, or None when the pixmap is null or has no pixels.
    Uses: focus peaking.
    Limitations: the conversion uses RGB888 and drops row padding past the image width.
    """
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
    """
    Convert a BGR image back into a pixmap.

    Inputs: image, a BGR array.
    Output: a pixmap copied from an RGB888 image.
    Uses: focus peaking.
    Limitations: the pixmap copies the pixels, so it does not keep a reference to the array buffer.
    """
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    qimage = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.shape[1] * 3, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimage.copy())
