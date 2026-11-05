"""Threshold, weights, and the optional subject-detection toggle.

``ControlPanel`` is the Grouping and Scoring sections. The getters read burst gap, sharpness, center, minimum frames, subfolders, subject, eye, and the model id. The setters restore those from settings without emitting a change. ``weights`` builds the score mix. ``refresh_models`` rewrites the model list with a ready or download mark.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
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


class ControlPanel(QWidget):
    threshold_changed = Signal(float)
    weights_changed = Signal()
    subject_toggled = Signal(bool)
    eye_toggled = Signal(bool)
    model_changed = Signal(str)
    minimum_changed = Signal(int)

    def __init__(self) -> None:
        """
        Build the grouping and scoring controls with their default values.

        Inputs: none.
        Output: no return value. Burst gap starts at 1.0 seconds, minimum at 2 frames, sharpness at 75, and center at 65.
        Uses: the control panel beside the burst list.
        Limitations: the panel emits signals when those controls change and does not scan or rescore by itself. The model list is filled from the models folder during construction.
        """
        super().__init__()
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

        grouping = QGroupBox("Grouping")
        grouping.setObjectName("sectionBox")
        grouping_layout = QVBoxLayout(grouping)
        grouping_layout.setSpacing(6)
        grouping_layout.addWidget(self._row("Burst gap", self._threshold))
        grouping_layout.addWidget(self._row("Minimum", self._minimum))
        grouping_layout.addWidget(self._recursive)

        scoring = QGroupBox("Scoring")
        scoring.setObjectName("sectionBox")
        scoring_layout = QVBoxLayout(scoring)
        scoring_layout.setSpacing(6)
        scoring_layout.addWidget(self._slider_row(self._sharpness_label, self._sharpness))
        scoring_layout.addWidget(self._slider_row(self._center_label, self._center))
        scoring_layout.addWidget(self._subject)
        scoring_layout.addWidget(self._eye)
        scoring_layout.addWidget(self._model)
        scoring_layout.addWidget(self._model_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(grouping)
        layout.addWidget(scoring)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        self._refresh_labels()

    def _row(self, label: str, widget: QWidget) -> QWidget:
        """
        Place a field label and a control on one row.

        Inputs: label, the field name; widget, the control.
        Output: a row widget with the label and the control.
        Uses: Burst gap and Minimum in the Grouping section.
        Limitations: the label's minimum width is 88 pixels.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        name = QLabel(label)
        name.setObjectName("fieldLabel")
        name.setMinimumWidth(88)
        layout.addWidget(name)
        layout.addWidget(widget, 1)
        return row

    def _slider_row(self, label: QLabel, slider: QSlider) -> QWidget:
        """
        Place a percentage label and a slider on one row.

        Inputs: label, the percentage label; slider, the slider.
        Output: a row widget with the label and the slider.
        Uses: Sharp and Center in the Scoring section.
        Limitations: the label's minimum width is 128 pixels.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label.setObjectName("fieldLabel")
        label.setMinimumWidth(128)
        layout.addWidget(label)
        layout.addWidget(slider, 1)
        return row

    def threshold(self) -> float:
        """
        Read the burst-gap setting.

        Inputs: none.
        Output: the burst gap in seconds.
        Uses: grouping for the next scan and for regrouping an open library.
        Limitations: the spin box accepts 0.0 through 30.0 seconds in steps of 0.1.
        """
        return float(self._threshold.value())

    def sharpness_percent(self) -> int:
        """
        Read the sharpness share of the score.

        Inputs: none.
        Output: an integer from 0 to 100.
        Uses: the Sharp slider in the Scoring section.
        Limitations: the slider range is 0 to 100. Exposure is the remaining share and is computed when weights are read.
        """
        return int(self._sharpness.value())

    def center_percent(self) -> int:
        """
        Read how much the middle of the frame counts toward sharpness.

        Inputs: none.
        Output: an integer from 0 to 100.
        Uses: the Center slider in the Scoring section.
        Limitations: the slider range is 0 to 100.
        """
        return int(self._center.value())

    def minimum_frames(self) -> int:
        """
        Read the minimum burst length that stays on the list.

        Inputs: none.
        Output: an integer frame count.
        Uses: hiding short bursts and leaving them out when keepers are saved.
        Limitations: the spin box accepts 1 through 99 frames.
        """
        return int(self._minimum.value())

    def set_minimum_frames(self, value: int) -> None:
        """
        Set the minimum burst length without emitting a change.

        Inputs: value, the frame count. Values below 1 become 1.
        Output: no return value. The spin box shows the clamped value.
        Uses: restoring the saved Minimum frames setting.
        Limitations: signals are blocked, so the burst list is not refreshed by this call.
        """
        self._minimum.blockSignals(True)
        self._minimum.setValue(max(1, int(value)))
        self._minimum.blockSignals(False)

    def recursive(self) -> bool:
        """
        Read whether the next scan includes subfolders.

        Inputs: none.
        Output: True when the checkbox is checked.
        Uses: the Grouping section and the scan worker.
        Limitations: checking the box does not start a scan.
        """
        return self._recursive.isChecked()

    def prefer_subject(self) -> bool:
        """
        Read whether sharpness is scored inside a detected animal.

        Inputs: none.
        Output: True when Subject-aware sharpness is checked.
        Uses: the Scoring section and the scan worker.
        Limitations: the checkbox state alone does not download a model.
        """
        return self._subject.isChecked()

    def eye_scoring(self) -> bool:
        """
        Read whether eye sharpness is requested.

        Inputs: none.
        Output: True when Eye sharpness is checked.
        Uses: the Scoring section and the scan worker.
        Limitations: the checkbox does not rescan by itself. The review window rescans when the toggle signal fires and a folder is open.
        """
        return self._eye.isChecked()

    def weights(self) -> Weights:
        """
        Build the scoring weights from the sliders and the subject checkbox.

        Inputs: none.
        Output: a Weights value. Sharpness and center are fractions of the slider percents. Exposure is the share left after sharpness, and never below zero.
        Uses: rescoring an open library and the next scan.
        Limitations: eye scoring is a separate checkbox and is not part of the returned Weights.
        """
        sharpness = self._sharpness.value() / 100.0
        return Weights(
            sharpness=sharpness,
            exposure=max(0.0, 1.0 - sharpness),
            center_mix=self._center.value() / 100.0,
            prefer_subject=self._subject.isChecked(),
        )

    def set_threshold(self, value: float) -> None:
        """
        Set the burst gap without emitting a change.

        Inputs: value, the gap in seconds.
        Output: no return value. The spin box shows that value.
        Uses: restoring a saved gap or matching a loaded library.
        Limitations: signals are blocked, so the open library is not regrouped by this call.
        """
        self._threshold.blockSignals(True)
        self._threshold.setValue(value)
        self._threshold.blockSignals(False)

    def set_sharpness_percent(self, value: int) -> None:
        """
        Set the sharpness share without emitting a change, then refresh its label.

        Inputs: value, an integer percent.
        Output: no return value. The slider and the Sharp label update.
        Uses: restoring the saved sharpness setting.
        Limitations: signals are blocked, so weights_changed is not emitted.
        """
        self._sharpness.blockSignals(True)
        self._sharpness.setValue(value)
        self._sharpness.blockSignals(False)
        self._refresh_labels()

    def set_center_percent(self, value: int) -> None:
        """
        Set the center share without emitting a change, then refresh its label.

        Inputs: value, an integer percent.
        Output: no return value. The slider and the Center label update.
        Uses: restoring the saved center setting.
        Limitations: signals are blocked, so weights_changed is not emitted.
        """
        self._center.blockSignals(True)
        self._center.setValue(value)
        self._center.blockSignals(False)
        self._refresh_labels()

    def set_recursive(self, checked: bool) -> None:
        """
        Set the subfolder checkbox.

        Inputs: checked, whether subfolders are included on the next scan.
        Output: no return value. The checkbox matches checked.
        Uses: restoring the saved subfolder setting.
        Limitations: signals are not blocked. The checkbox has no scan slot of its own.
        """
        self._recursive.setChecked(checked)

    def set_subject_checked(self, checked: bool) -> None:
        """
        Set the subject checkbox without emitting a change.

        Inputs: checked, whether subject-aware sharpness is on.
        Output: no return value. The checkbox matches checked.
        Uses: restoring the setting, and turning it off when a model download fails or is cancelled.
        Limitations: signals are blocked, so this call does not start a download or a scan.
        """
        self._subject.blockSignals(True)
        self._subject.setChecked(checked)
        self._subject.blockSignals(False)

    def set_eye_scoring(self, checked: bool) -> None:
        """
        Set the eye-sharpness checkbox without emitting a change.

        Inputs: checked, whether eye sharpness is on.
        Output: no return value. The checkbox matches checked.
        Uses: restoring the saved eye-scoring setting.
        Limitations: signals are blocked, so this call does not rescan.
        """
        self._eye.blockSignals(True)
        self._eye.setChecked(checked)
        self._eye.blockSignals(False)

    def model_id(self) -> str:
        """
        Read the selected subject model id.

        Inputs: none.
        Output: the combo item's string data, or the default model id when the current data is missing or empty.
        Uses: the subject model combo in the Scoring section.
        Limitations: an empty combo still returns the default model id.
        """
        value = self._model.currentData()
        if isinstance(value, str) and value:
            return value
        return DEFAULT_MODEL_ID

    def model_ids(self) -> tuple[str, ...]:
        """
        Read every model id currently listed.

        Inputs: none.
        Output: the item-data strings, skipping empty entries.
        Uses: the subject model combo.
        Limitations: the visible labels and the ready or download state are not included.
        """
        return tuple(
            str(self._model.itemData(index))
            for index in range(self._model.count())
            if self._model.itemData(index)
        )

    def set_model_id(self, model_id: str) -> None:
        """
        Select a model by id without emitting a change.

        Inputs: model_id, the id to select.
        Output: no return value. When the id is listed, the combo selects it and the hint updates.
        Uses: restoring the saved subject model.
        Limitations: an id that is not in the combo is ignored. Signals are blocked on a successful selection.
        """
        index = self._model.findData(model_id)
        if index < 0:
            return
        self._model.blockSignals(True)
        self._model.setCurrentIndex(index)
        self._model.blockSignals(False)
        self._update_model_hint()

    def set_model_enabled(self, enabled: bool) -> None:
        """
        Enable or disable the subject model combo.

        Inputs: enabled, whether the photographer can change the model.
        Output: no return value. The combo's enabled state changes.
        Uses: locking the combo while a model downloads.
        Limitations: the subject and eye checkboxes are left as they are.
        """
        self._model.setEnabled(enabled)

    def refresh_models(self) -> None:
        """
        Rebuild the model list from the models folder.

        Inputs: none.
        Output: no return value. Each model is labeled ready or download, and the current id is kept when it still exists.
        Uses: the subject model combo after a download.
        Limitations: model_changed is not emitted, because signals are blocked while the items are rebuilt.
        """
        self._rebuild_model_items()

    def _emit_model(self, _index: int) -> None:
        """
        Refresh the model hint and announce the selected model.

        Inputs: _index, the combo index. The index is unused.
        Output: no return value. Emits model_changed with the current model id.
        Uses: choosing a subject model in the Scoring section.
        Limitations: the emitted id is read from the combo after the change. The index argument is ignored.
        """
        self._update_model_hint()
        self.model_changed.emit(self.model_id())

    def _rebuild_model_items(self) -> None:
        """
        Fill the model combo from the known models and the files on disk.

        Inputs: none.
        Output: no return value. Each item shows the label plus ready or download, with a tooltip of the model note and file path.
        Uses: the subject model combo.
        Limitations: signals are blocked. A current id that is no longer listed falls back to the first item. The hint is updated before this returns.
        """
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
        """
        Describe whether the selected model file is already in the models folder.

        Inputs: none.
        Output: no return value. A ready file names the filename. A missing file says Rescan downloads it. An unknown id shows the models folder path.
        Uses: the hint under the subject model combo.
        Limitations: the hint tooltip is the model path when the id is known, and is left unchanged when the id is unknown.
        """
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
        self._model_label.setText("Rescan downloads this model if it is missing.")

    def _slider(self, value: int, tip: str) -> QSlider:
        """
        Build a horizontal slider from 0 to 100.

        Inputs: value, the starting position; tip, the tooltip.
        Output: the slider.
        Uses: the Sharp and Center controls.
        Limitations: the range is fixed at 0 to 100.
        """
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(value)
        slider.setToolTip(tip)
        return slider

    def _emit_weights(self) -> None:
        """
        Announce that the sharpness or center slider moved.

        Inputs: none.
        Output: no return value. Emits weights_changed.
        Uses: rescoring an open library from the Scoring section.
        Limitations: the signal carries no value. The window reads the sliders itself.
        """
        self.weights_changed.emit()

    def _refresh_labels(self) -> None:
        """
        Show the current sharpness and center percentages on their labels.

        Inputs: none.
        Output: no return value. The Sharp label shows the percent, and its tooltip states the exposure share. The Center label shows its percent.
        Uses: the Scoring section labels.
        Limitations: the center tooltip is not changed here. Exposure in the sharp tooltip is 100 minus the sharpness percent.
        """
        sharpness = self._sharpness.value()
        self._sharpness_label.setText(f"Sharp {sharpness}%")
        self._sharpness_label.setToolTip(
            f"Sharpness is {sharpness}% of the score. Exposure is the other {100 - sharpness}%."
        )
        self._center_label.setText(f"Center {self._center.value()}%")
