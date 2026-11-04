"""Burst list helpers and the horizontal filmstrip."""

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


class ThumbCard(QFrame):
    clicked = Signal(int)

    def __init__(self, index: int, thumb: Thumb) -> None:
        super().__init__()
        self.index = index
        self._best = thumb.best
        self._decision = thumb.decision
        self._selected = thumb.selected
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
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._apply_style()

    def set_flags(self, *, best: bool, decision: str | None, badge: str, score_text: str) -> None:
        self._best = best
        self._decision = decision
        self.badge.setText(badge)
        self.score.setText(score_text)
        self._apply_style()

    def _apply_style(self) -> None:
        if self._selected and self._best:
            color = "#7dffc3"
        elif self._selected:
            color = "#79b8ff"
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

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("filmstrip")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setMinimumHeight(214)
        self._inner = QWidget()
        self._layout = QHBoxLayout(self._inner)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self._layout.addStretch(1)
        self.setWidget(self._inner)
        self._cards: list[ThumbCard] = []

    def set_cards(self, cards: list[Thumb]) -> None:
        self._clear()
        for index, thumb in enumerate(cards):
            card = ThumbCard(index, thumb)
            card.clicked.connect(self.selected.emit)
            self._layout.insertWidget(self._layout.count() - 1, card)
            self._cards.append(card)
        selected = next((index for index, thumb in enumerate(cards) if thumb.selected), None)
        if selected is not None:
            self.mark_selected(selected)

    def mark_selected(self, index: int) -> None:
        for card_index, card in enumerate(self._cards):
            card.set_selected(card_index == index)
        if 0 <= index < len(self._cards):
            self.ensureWidgetVisible(self._cards[index], 24, 0)

    def refresh_flags(self, cards: list[Thumb]) -> None:
        if len(cards) != len(self._cards):
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

    def wheelEvent(self, event) -> None:  # type: ignore[no-untyped-def]
        delta = event.angleDelta().y() or event.angleDelta().x()
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - delta)
        event.accept()

    def _clear(self) -> None:
        while self._layout.count() > 1:
            item = self._layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards = []
