"""Threshold, weights, and the optional subject-detection toggle."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from outburst.models import Weights


class ControlPanel(QGroupBox):
    threshold_changed = Signal(float)
    weights_changed = Signal()
    subject_toggled = Signal(bool)
    choose_model = Signal()

    def __init__(self) -> None:
        super().__init__("Scoring")
        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.0, 30.0)
        self._threshold.setSingleStep(0.1)
        self._threshold.setDecimals(1)
        self._threshold.setValue(1.0)
        self._threshold.setSuffix(" s")
        self._threshold.setToolTip("Frames closer together than this stay in one burst.")
        self._threshold.valueChanged.connect(lambda value: self.threshold_changed.emit(float(value)))

        self._sharpness = self._slider(75, "Share of the score that comes from sharpness. The rest is exposure.")
        self._center = self._slider(65, "How much the middle of the frame counts toward sharpness.")
        self._sharpness.valueChanged.connect(self._emit_weights)
        self._center.valueChanged.connect(self._emit_weights)

        self._sharpness_label = QLabel()
        self._center_label = QLabel()
        self._sharpness.valueChanged.connect(self._refresh_labels)
        self._center.valueChanged.connect(self._refresh_labels)

        self._recursive = QCheckBox("Include subfolders on the next scan")
        self._subject = QCheckBox("Subject-aware sharpness")
        self._subject.setToolTip(
            "Score sharpness inside a detected animal. Requires a local YOLO .pt file. "
            "Outburst does not download one."
        )
        self._subject.toggled.connect(self.subject_toggled.emit)
        self._model_button = QPushButton("Choose model…")
        self._model_button.clicked.connect(self.choose_model.emit)
        self._model_label = QLabel("No model selected")
        self._model_label.setObjectName("hint")
        self._model_label.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Burst gap"))
        layout.addWidget(self._threshold)
        layout.addWidget(self._sharpness_label)
        layout.addWidget(self._sharpness)
        layout.addWidget(self._center_label)
        layout.addWidget(self._center)
        layout.addWidget(self._recursive)
        layout.addWidget(self._subject)
        row = QHBoxLayout()
        row.addWidget(self._model_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addWidget(self._model_label)
        hint = QLabel("Sharpness and exposure update the ranking immediately.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self._refresh_labels()

    def threshold(self) -> float:
        return float(self._threshold.value())

    def sharpness_percent(self) -> int:
        return int(self._sharpness.value())

    def center_percent(self) -> int:
        return int(self._center.value())

    def recursive(self) -> bool:
        return self._recursive.isChecked()

    def prefer_subject(self) -> bool:
        return self._subject.isChecked()

    def weights(self) -> Weights:
        sharpness = self._sharpness.value() / 100.0
        return Weights(
            sharpness=sharpness,
            exposure=max(0.0, 1.0 - sharpness),
            center_mix=self._center.value() / 100.0,
            prefer_subject=self._subject.isChecked(),
        )

    def set_threshold(self, value: float) -> None:
        self._threshold.blockSignals(True)
        self._threshold.setValue(value)
        self._threshold.blockSignals(False)

    def set_sharpness_percent(self, value: int) -> None:
        self._sharpness.blockSignals(True)
        self._sharpness.setValue(value)
        self._sharpness.blockSignals(False)
        self._refresh_labels()

    def set_center_percent(self, value: int) -> None:
        self._center.blockSignals(True)
        self._center.setValue(value)
        self._center.blockSignals(False)
        self._refresh_labels()

    def set_recursive(self, checked: bool) -> None:
        self._recursive.setChecked(checked)

    def set_subject_checked(self, checked: bool) -> None:
        self._subject.blockSignals(True)
        self._subject.setChecked(checked)
        self._subject.blockSignals(False)

    def set_model_name(self, name: str) -> None:
        self._model_label.setText(name)

    def _slider(self, value: int, tip: str) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(value)
        slider.setToolTip(tip)
        return slider

    def _emit_weights(self) -> None:
        self.weights_changed.emit()

    def _refresh_labels(self) -> None:
        sharpness = self._sharpness.value()
        self._sharpness_label.setText(f"Sharpness {sharpness}%  ·  exposure {100 - sharpness}%")
        self._center_label.setText(f"Center weight {self._center.value()}%")
