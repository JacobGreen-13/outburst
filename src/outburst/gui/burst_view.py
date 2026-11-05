"""Burst list helpers and the horizontal filmstrip.

``Thumb`` is one card's data, including the burst frame index. ``ThumbCard`` draws it and emits a click with the Shift state. ``Filmstrip.set_cards``, ``mark_selected``, and ``mark_compared`` show the row. ``refresh_flags`` updates badges, or rebuilds the row when the order changed. The wheel scrolls the strip and does not zoom the preview.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


@dataclass(frozen=True)
class Thumb:
    title: str
    score_text: str
    badge: str
    thumb_path: Path | None
    selected: bool
    best: bool
    decision: str | None
    tooltip: str
    frame_index: int
    compared: bool = False


class ThumbCard(QFrame):
    clicked = Signal(int, bool)

    def __init__(self, index: int, thumb: Thumb) -> None:
        """
        Build one filmstrip card from a thumbnail description.

        Inputs: index, the shooting-order frame index this card represents; thumb, the title, score, badge, image path, and flags.
        Output: no return value. The card shows the scaled thumbnail, or the text "No preview", and paints its border.
        Uses: one frame in the filmstrip.
        Limitations: the thumbnail is scaled to 164 by 108 from the embedded preview path. A null pixmap shows "No preview".
        """
        super().__init__()
        self.index = index
        self._best = thumb.best
        self._decision = thumb.decision
        self._selected = thumb.selected
        self._compared = thumb.compared
        self.setObjectName("thumbCard")
        self.setFixedWidth(176)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip(thumb.tooltip)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(3)
        self.image = QLabel()
        self.image.setFixedSize(164, 108)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = QPixmap(str(thumb.thumb_path)) if thumb.thumb_path else QPixmap()
        if pixmap.isNull():
            self.image.setText("No preview")
        else:
            self.image.setPixmap(
                pixmap.scaled(
                    164,
                    108,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.title = QLabel(thumb.title)
        self.score = QLabel(thumb.score_text)
        self.badge = QLabel(thumb.badge)
        for label in (self.title, self.score, self.badge):
            label.setWordWrap(False)
        layout.addWidget(self.image)
        layout.addWidget(self.title)
        layout.addWidget(self.score)
        layout.addWidget(self.badge)
        self._apply_style()

    def mousePressEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Turn a left click into a card signal, including Shift.

        Inputs: event, the mouse press.
        Output: no return value. A left click emits clicked with the frame index and whether Shift is held, then the base handler runs.
        Uses: selecting a filmstrip frame, or Shift-click to compare it.
        Limitations: only the left button emits the signal.
        """
        if event.button() == Qt.MouseButton.LeftButton:
            shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.clicked.emit(self.index, shift)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        """
        Show or hide the selection border on this card.

        Inputs: selected, whether this card is the current frame.
        Output: no return value. The card border is redrawn.
        Uses: the filmstrip selection.
        Limitations: this updates the card only. The filmstrip scrolls the selected card into view.
        """
        self._selected = selected
        self._apply_style()

    def set_compared(self, compared: bool) -> None:
        """
        Show or hide the compare border on this card.

        Inputs: compared, whether this card is the other side of compare.
        Output: no return value. The card border is redrawn.
        Uses: the compare highlight in the filmstrip.
        Limitations: a selected card keeps the selection color. The compare color is used when the card is not selected.
        """
        self._compared = compared
        self._apply_style()

    def set_flags(self, *, best: bool, decision: str | None, badge: str, score_text: str) -> None:
        """
        Update the best, decision, badge, and score text without rebuilding the card.

        Inputs: best; decision, "keep", "reject", or None; badge; score_text.
        Output: no return value. The badge and score labels change and the border is redrawn.
        Uses: refreshing filmstrip marks after a decision or a rescore.
        Limitations: the thumbnail image and the title stay as they were.
        """
        self._best = best
        self._decision = decision
        self.badge.setText(badge)
        self.score.setText(score_text)
        self._apply_style()

    def _apply_style(self) -> None:
        """
        Paint the card border and label colors from its current flags.

        Inputs: none.
        Output: no return value. The frame stylesheet and the badge color are replaced.
        Uses: the filmstrip card appearance.
        Limitations: selection is green when the card is also the best frame and blue otherwise. Compare gold, reject red, and keeper green apply only when the card is not selected.
        """
        if self._selected and self._best:
            color = "#7dffc3"
        elif self._selected:
            color = "#79b8ff"
        elif self._compared:
            color = "#e6c15a"
        elif self._decision == "reject":
            color = "#8a3a3a"
        elif self._decision == "keep" or self._best:
            color = "#1f8a5b"
        else:
            color = "#3a3a3a"
        title_color = "#8d8d8d" if self._decision == "reject" else "#f2f2f2"
        badge_color = "#e06c6c" if self._decision == "reject" else "#7dffc3"
        self.setStyleSheet(
            "QFrame#thumbCard {"
            f"background: #242424; border: 2px solid {color}; border-radius: 8px;"
            "}"
            f"QLabel {{ color: {title_color}; background: transparent; }}"
        )
        self.badge.setStyleSheet(f"color: {badge_color}; background: transparent; font-weight: 600;")
        self.score.setStyleSheet("color: #f2f2f2; background: transparent; font-weight: 600;")


class Filmstrip(QScrollArea):
    """Contact-sheet row for the frames inside one burst."""

    selected = Signal(int)
    compare_chosen = Signal(int)

    def __init__(self) -> None:
        """
        Build an empty horizontal filmstrip.

        Inputs: none.
        Output: no return value. A scroll area with a trailing stretch is ready for cards.
        Uses: the filmstrip under the preview.
        Limitations: the vertical scrollbar stays off. The minimum height is 196 pixels, so extra frames scroll sideways.
        """
        super().__init__()
        self.setObjectName("filmstrip")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(196)
        self._inner = QWidget()
        self._layout = QHBoxLayout(self._inner)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self.setWidget(self._inner)
        self._cards: list[ThumbCard] = []

    def set_cards(self, cards: list[Thumb]) -> None:
        """
        Replace the filmstrip with a new row of cards.

        Inputs: cards, thumbnail descriptions in display order.
        Output: no return value. Old cards are removed, new cards are inserted, and a selected card is scrolled into view.
        Uses: showing one burst in the filmstrip.
        Limitations: the cards are shown in the order given. Time is shooting order and Score puts the highest composite first with ties in shooting order; that order is chosen by the caller.
        """
        self._clear()
        for thumb in cards:
            card = ThumbCard(thumb.frame_index, thumb)
            card.clicked.connect(self._on_card)
            self._layout.insertWidget(self._layout.count() - 1, card)
            self._cards.append(card)
        selected = next((thumb.frame_index for thumb in cards if thumb.selected), None)
        if selected is not None:
            self.mark_selected(selected)

    def mark_selected(self, index: int) -> None:
        """
        Mark one card as the current frame and scroll it into view.

        Inputs: index, the shooting-order frame index to select.
        Output: no return value. Each card's selection flag is updated, and a matching card is kept visible.
        Uses: the filmstrip selection.
        Limitations: an index that matches no card clears every selection and does not scroll.
        """
        chosen: ThumbCard | None = None
        for card in self._cards:
            selected = card.index == index
            card.set_selected(selected)
            if selected:
                chosen = card
        if chosen is not None:
            self.ensureWidgetVisible(chosen, 24, 0)

    def mark_compared(self, index: int | None) -> None:
        """
        Mark the card that is the other side of compare.

        Inputs: index, the shooting-order frame index to highlight, or None to clear the highlight.
        Output: no return value. Each card's compare flag is updated.
        Uses: the compare highlight in the filmstrip.
        Limitations: None clears every compare highlight, including when compare is turned off.
        """
        for card in self._cards:
            card.set_compared(index is not None and card.index == index)

    def _on_card(self, index: int, shift: bool) -> None:
        """
        Route a card click to compare or to selection.

        Inputs: index, the shooting-order frame index; shift, whether Shift was held.
        Output: no return value. Shift emits compare_chosen. A click without Shift emits selected.
        Uses: filmstrip clicks.
        Limitations: a Shift-click emits only the compare signal.
        """
        if shift:
            self.compare_chosen.emit(index)
            return
        self.selected.emit(index)

    def refresh_flags(self, cards: list[Thumb]) -> None:
        """
        Update card marks, or rebuild the row when the frame order changed.

        Inputs: cards, thumbnail descriptions in the current display order.
        Output: no return value. Flags, selection, and compare state update in place when the indexes match. A different index order rebuilds the row.
        Uses: keeper, reject, star, and score updates on the filmstrip.
        Limitations: a changed order deletes the existing cards and builds new ones, so thumbnail images are loaded again.
        """
        incoming = [thumb.frame_index for thumb in cards]
        if incoming != [card.index for card in self._cards]:
            self.set_cards(cards)
            return
        for card, thumb in zip(self._cards, cards, strict=True):
            card.set_flags(
                best=thumb.best,
                decision=thumb.decision,
                badge=thumb.badge,
                score_text=thumb.score_text,
            )
            card.set_selected(thumb.selected)
            card.set_compared(thumb.compared)

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        """
        Scroll the filmstrip sideways with the mouse wheel.

        Inputs: event, the wheel event.
        Output: no return value. The horizontal scrollbar moves by the wheel delta and the event is accepted.
        Uses: the filmstrip.
        Limitations: a vertical wheel delta is used when it is non-zero; otherwise the horizontal delta is used. The scrollbar moves by the opposite of that delta.
        """
        delta = event.angleDelta().y() or event.angleDelta().x()
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - delta)
        event.accept()

    def _clear(self) -> None:
        """
        Delete every card and leave the trailing stretch.

        Inputs: none.
        Output: no return value. Card widgets are scheduled for deletion and the card list is emptied.
        Uses: replacing the filmstrip contents.
        Limitations: the stretch item at the end of the layout is kept.
        """
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards = []
