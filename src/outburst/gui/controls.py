"""Threshold, weights, and the optional subject-detection toggle."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from outburst.models import Weights
from outburst.subject import (
    DEFAULT_MODEL_ID,
    MODEL_OPTIONS,
    model_is_ready,
    model_path,
    models_dir,
    option_by_id,
)


class ControlPanel(QGroupBox):
    threshold_changed = Signal(float)
    weights_changed = Signal()
    subject_toggled = Signal(bool)
    eye_toggled = Signal(bool)
    model_changed = Signal(str)
    minimum_changed = Signal(int)

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

        self._minimum = QSpinBox()
        self._minimum.setRange(1, 99)
        self._minimum.setValue(2)
        self._minimum.setSuffix(" frames")
        self._minimum.setToolTip(
            "Bursts with fewer frames than this stay off the list and are left out when you save keepers."
        )
        self._minimum.valueChanged.connect(lambda value: self.minimum_changed.emit(int(value)))
        self._recursive = QCheckBox("Include subfolders on the next scan")
        self._subject = QCheckBox("Subject-aware sharpness")
        self._subject.setToolTip(
            "Score sharpness inside a detected animal. "
            "Choosing a model turns this on. Rescan downloads that model if it is not in the models folder."
        )
        self._subject.toggled.connect(self.subject_toggled.emit)
        self._eye = QCheckBox("Eye sharpness")
        self._eye.setToolTip(
            "Rate a dark, round eye when it is large enough in the preview. "
            "In a burst with two or more rated eyes, the sharpest eye becomes the best frame "
            "and is marked a keeper. A smaller eye is noted and left out of the ranking. "
            "With subject scoring on, the search stays inside the animal. Turning this on or off rescans."
        )
        self._eye.toggled.connect(self.eye_toggled.emit)
        self._model = QComboBox()
        self._model.setToolTip(
            "Subject model. Choosing one turns subject scoring on. "
            "Rescan saves a model marked download into the models folder and scores with it."
        )
        self._model_label = QLabel()
        self._model_label.setObjectName("hint")
        self._model_label.setWordWrap(True)
        self._model_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._rebuild_model_items()
        self._model.currentIndexChanged.connect(lambda index: self._emit_model(int(index)))

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Burst gap"))
        layout.addWidget(self._threshold)
        layout.addWidget(self._sharpness_label)
        layout.addWidget(self._sharpness)
        layout.addWidget(self._center_label)
        layout.addWidget(self._center)
        layout.addWidget(QLabel("Minimum frames"))
        layout.addWidget(self._minimum)
        layout.addWidget(self._recursive)
        layout.addWidget(self._subject)
        layout.addWidget(self._eye)
        layout.addWidget(self._model)
        layout.addWidget(self._model_label)
        hint = QLabel("Sharpness and exposure update the ranking immediately.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        hint.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        layout.addWidget(hint)
        self._refresh_labels()

    def threshold(self) -> float:
        return float(self._threshold.value())

    def sharpness_percent(self) -> int:
        return int(self._sharpness.value())

    def center_percent(self) -> int:
        return int(self._center.value())

    def minimum_frames(self) -> int:
        return int(self._minimum.value())

    def set_minimum_frames(self, value: int) -> None:
        self._minimum.blockSignals(True)
        self._minimum.setValue(max(1, int(value)))
        self._minimum.blockSignals(False)

    def recursive(self) -> bool:
        return self._recursive.isChecked()

    def prefer_subject(self) -> bool:
        return self._subject.isChecked()

    def eye_scoring(self) -> bool:
        return self._eye.isChecked()

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

    def set_eye_scoring(self, checked: bool) -> None:
        self._eye.blockSignals(True)
        self._eye.setChecked(checked)
        self._eye.blockSignals(False)

    def model_id(self) -> str:
        value = self._model.currentData()
        if isinstance(value, str) and value:
            return value
        return DEFAULT_MODEL_ID

    def model_ids(self) -> tuple[str, ...]:
        return tuple(
            str(self._model.itemData(index))
            for index in range(self._model.count())
            if self._model.itemData(index)
        )

    def set_model_id(self, model_id: str) -> None:
        index = self._model.findData(model_id)
        if index < 0:
            return
        self._model.blockSignals(True)
        self._model.setCurrentIndex(index)
        self._model.blockSignals(False)
        self._update_model_hint()

    def set_model_enabled(self, enabled: bool) -> None:
        self._model.setEnabled(enabled)

    def refresh_models(self) -> None:
        self._rebuild_model_items()

    def _emit_model(self, _index: int) -> None:
        self._update_model_hint()
        self.model_changed.emit(self.model_id())

    def _rebuild_model_items(self) -> None:
        current = self.model_id() if self._model.count() else DEFAULT_MODEL_ID
        directory = models_dir()
        self._model.blockSignals(True)
        self._model.clear()
        for option in MODEL_OPTIONS:
            ready = model_is_ready(model_path(option, directory))
            state = "ready" if ready else "download"
            self._model.addItem(f"{option.label}  ·  {state}", option.id)
            self._model.setItemData(
                self._model.count() - 1,
                f"{option.note}\n{directory / option.filename}",
                Qt.ItemDataRole.ToolTipRole,
            )
        index = self._model.findData(current)
        self._model.setCurrentIndex(index if index >= 0 else 0)
        self._model.blockSignals(False)
        self._update_model_hint()

    def _update_model_hint(self) -> None:
        option = option_by_id(self.model_id())
        directory = models_dir()
        if option is None:
            self._model_label.setText(str(directory))
            return
        path = model_path(option, directory)
        self._model_label.setToolTip(str(path))
        if model_is_ready(path):
            self._model_label.setText(f"{option.filename} is in the models folder.")
            return
        self._model_label.setText(
            "Choosing this model turns subject scoring on. Rescan downloads it if it is not in the models folder."
        )

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
