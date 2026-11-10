"""Threshold, weights, and the optional subject-detection toggle.

``ControlPanel`` is the Grouping and Scoring sections. Grouping starts collapsed and Scoring starts open. Subject and bird eye controls stay hidden until those modules are installed, and an install button is shown in their place. The getters read burst gap, sharpness, center, minimum frames, minimum score, subfolders, subject, bird eye detection, the subject-box and eye-circle marks, the model id, and whether each section is open. The setters restore those from settings without emitting a change. ``weights`` builds the score mix. ``refresh_models`` rewrites the model list with a ready or download mark. ``set_score_effect`` shows how many frames Minimum score is leaving on screen. ``set_detection_installed`` swaps the install button for the detection controls.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
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
    subject_box_toggled = Signal(bool)
    eye_circle_toggled = Signal(bool)
    model_changed = Signal(str)
    minimum_changed = Signal(int)
    minimum_score_changed = Signal()
    install_detection_clicked = Signal()

    def __init__(self) -> None:
        """
        Build the grouping and scoring controls with their default values.

        Inputs: none.
        Output: no return value. Burst gap starts at 1.0 seconds, minimum at 2 frames, sharpness at 75, center at 65, and minimum score off at 50. Grouping starts collapsed and Scoring starts open.
        Uses: the control panel beside the burst list.
        Limitations: the panel emits signals when those controls change and does not scan or rescore by itself. The model list is filled from the models folder during construction. Opening or closing a section only shows or hides its controls.
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
        self._score_filter = QCheckBox("Minimum score")
        self._score_filter.setToolTip(
            "Hide frames whose score is below this number. "
            "A burst with no frame at or above it leaves the list. "
            "Keeper and star marks on hidden frames stay, and Save keepers still writes them."
        )
        self._score_minimum = QSpinBox()
        self._score_minimum.setRange(0, 100)
        self._score_minimum.setValue(50)
        self._score_minimum.setEnabled(False)
        self._score_minimum.setToolTip("The same 0 to 100 score shown on the filmstrip. A frame labeled 50 stays visible at 50.")
        self._score_filter.toggled.connect(self._on_score_filter_toggled)
        self._score_minimum.valueChanged.connect(lambda _value: self.minimum_score_changed.emit())
        self._recursive = QCheckBox("Include subfolders on the next scan")
        self._subject = QCheckBox("Subject-aware sharpness")
        self._subject.setToolTip(
            "Score sharpness inside a detected animal. "
            "The score uses that box when an animal is found, and the whole frame when it is not."
        )
        self._subject.toggled.connect(self.subject_toggled.emit)
        self._show_box = QCheckBox("Show subject box")
        self._show_box.setChecked(True)
        self._show_box.setToolTip(
            "Draw the detected animal box on the preview and the compare pane. "
            "This does not rescan and does not turn subject scoring off."
        )
        self._show_box.toggled.connect(self.subject_box_toggled.emit)
        self._eye = QCheckBox("Bird eye detection")
        self._eye.setToolTip(
            "Use the open-weight bird eye model. "
            "In a burst with two or more rated eyes, the sharpest eye becomes the best frame "
            "and is marked a keeper. The model is birds only. It stays on this computer."
        )
        self._eye.toggled.connect(self.eye_toggled.emit)
        self._show_eye = QCheckBox("Show eye circle")
        self._show_eye.setChecked(True)
        self._show_eye.setToolTip(
            "Draw the bird eye ring on the preview and the compare pane. "
            "This does not rescan and does not turn bird eye detection off."
        )
        self._show_eye.toggled.connect(self.eye_circle_toggled.emit)
        self._model = QComboBox()
        self._model.setToolTip(
            "Subject model. Choosing one turns subject scoring on. "
            "A model marked download is saved into the models folder."
        )
        self._model_label = QLabel()
        self._model_label.setObjectName("hint")
        self._model_label.setWordWrap(True)
        self._model_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._rebuild_model_items()
        self._model.currentIndexChanged.connect(lambda index: self._emit_model(int(index)))

        self._install_detection = QPushButton("Install subject and bird eye detection")
        self._install_detection.setObjectName("installDetection")
        self._install_detection.setToolTip(
            "Check this computer and the internet connection, then accept the license. "
            "Outburst installs subject and bird eye detection and downloads the models after that."
        )
        self._install_detection.clicked.connect(self.install_detection_clicked.emit)
        self._detection = QWidget()
        self._detection.setObjectName("detectionControls")
        detection_layout = QVBoxLayout(self._detection)
        detection_layout.setContentsMargins(0, 0, 0, 0)
        detection_layout.setSpacing(6)
        detection_layout.addWidget(self._subject)
        detection_layout.addWidget(self._indented(self._show_box))
        detection_layout.addWidget(self._eye)
        detection_layout.addWidget(self._indented(self._show_eye))
        detection_layout.addWidget(self._model)
        detection_layout.addWidget(self._model_label)
        self._detection.hide()

        self._score_effect = QLabel("")
        self._score_effect.setObjectName("hint")
        self._score_effect.hide()

        self._grouping_body = QWidget()
        self._grouping_body.setObjectName("sectionBody")
        grouping_layout = QVBoxLayout(self._grouping_body)
        grouping_layout.setContentsMargins(8, 8, 8, 8)
        grouping_layout.setSpacing(6)
        grouping_layout.addWidget(self._row("Burst gap", self._threshold))
        grouping_layout.addWidget(self._row("Minimum", self._minimum))
        grouping_layout.addWidget(self._recursive)
        self._grouping_toggle = self._disclosure("Grouping", self._grouping_body, open_=False)

        self._scoring_body = QWidget()
        self._scoring_body.setObjectName("sectionBody")
        scoring_layout = QVBoxLayout(self._scoring_body)
        scoring_layout.setContentsMargins(8, 8, 8, 8)
        scoring_layout.setSpacing(6)
        scoring_layout.addWidget(self._slider_row(self._sharpness_label, self._sharpness))
        scoring_layout.addWidget(self._slider_row(self._center_label, self._center))
        scoring_layout.addWidget(self._score_row())
        scoring_layout.addWidget(self._install_detection)
        scoring_layout.addWidget(self._detection)
        self._scoring_toggle = self._disclosure("Scoring", self._scoring_body, open_=True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._grouping_toggle)
        layout.addWidget(self._grouping_body)
        layout.addWidget(self._scoring_toggle)
        layout.addWidget(self._scoring_body)
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

    def _indented(self, widget: QWidget) -> QWidget:
        """
        Place a checkbox under the detection option it belongs to.

        Inputs: widget, the show-box or show-circle checkbox.
        Output: a row with 22 pixels of space on the left.
        Uses: Show subject box and Show eye circle in the Scoring section.
        Limitations: the indent is visual. The checkbox has its own signal and does not change the detection checkbox.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(22, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(widget)
        return row

    def _score_row(self) -> QWidget:
        """
        Place the minimum-score checkbox and its number on one row.

        Inputs: none. The checkbox and spin box were created with the panel.
        Output: a row widget with the toggle and the 0 to 100 cutoff.
        Uses: Minimum score in the Scoring section.
        Limitations: the number stays disabled until the checkbox is on. The effect line stays hidden until the window reports a count.
        """
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(self._score_filter, 1)
        layout.addWidget(self._score_minimum)
        layout.addWidget(self._score_effect)
        return row

    def _disclosure(self, title: str, body: QWidget, *, open_: bool) -> QToolButton:
        """
        Build a header that shows or hides one section.

        Inputs: title, the section name; body, the controls to show or hide; open_, whether the section starts open.
        Output: the header button. The body is visible only when open_ is true.
        Uses: Grouping and Scoring in the control panel.
        Limitations: the button's arrow follows the open state. This does not change the values inside the section.
        """
        button = QToolButton()
        button.setObjectName("sectionToggle")
        button.setText(title)
        button.setCheckable(True)
        button.setChecked(open_)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setToolTip(f"Show or hide {title}.")
        body.setVisible(open_)
        button.toggled.connect(
            lambda checked, control=button, target=body: self._apply_section(control, target, checked)
        )
        return button

    def _apply_section(self, button: QToolButton, body: QWidget, open_: bool) -> None:
        """
        Show or hide a section and point its arrow the matching way.

        Inputs: button, the section header; body, the controls; open_, whether the section is open.
        Output: no return value. The body is visible when open_ is true, and the arrow points down when open and right when closed.
        Uses: the Grouping and Scoring headers.
        Limitations: this does not change the values inside the section and does not rescore.
        """
        button.setArrowType(Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow)
        body.setVisible(open_)

    def _on_score_filter_toggled(self, checked: bool) -> None:
        """
        Enable the score cutoff and tell the window the filter changed.

        Inputs: checked, whether Minimum score is now on.
        Output: no return value. The spin box follows the checkbox, and minimum_score_changed is emitted.
        Uses: the Minimum score checkbox.
        Limitations: the spin box value is kept while the filter is off, and this method does not hide frames itself.
        """
        self._score_minimum.setEnabled(checked)
        self.minimum_score_changed.emit()

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

    def minimum_score_enabled(self) -> bool:
        """
        Read whether frames below the minimum score are hidden.

        Inputs: none.
        Output: True when Minimum score is checked.
        Uses: the Scoring section and the review window.
        Limitations: a checked box with a cutoff of 0 still hides a frame that has no score.
        """
        return self._score_filter.isChecked()

    def minimum_score(self) -> int:
        """
        Read the minimum score cutoff.

        Inputs: none.
        Output: an integer from 0 to 100, even when the filter is off.
        Uses: the Minimum score spin box.
        Limitations: the spin box accepts 0 through 100. The filter uses this number only while the checkbox is on.
        """
        return int(self._score_minimum.value())

    def score_floor(self) -> int | None:
        """
        Read the active score cutoff.

        Inputs: none.
        Output: the minimum score when the checkbox is on, otherwise None.
        Uses: deciding which frames stay on the filmstrip.
        Limitations: None means every frame stays visible. The returned number is the display score from 0 to 100.
        """
        if not self._score_filter.isChecked():
            return None
        return self.minimum_score()

    def set_score_effect(self, showing: int | None, eligible: int | None) -> None:
        """
        Show how many frames Minimum score is leaving on screen.

        Inputs: showing, the frames still visible, or None to hide the line; eligible, the frames in bursts long enough to review.
        Output: no return value. The line reads "N of M showing" when both numbers are given, and is hidden otherwise.
        Uses: the Minimum score row while a library is open.
        Limitations: a missing number hides the line. This does not change the cutoff or hide frames itself.
        """
        if showing is None or eligible is None:
            self._score_effect.clear()
            self._score_effect.hide()
            return
        self._score_effect.setText(f"{showing} of {eligible} showing")
        self._score_effect.show()

    def grouping_open(self) -> bool:
        """
        Read whether the Grouping section is open.

        Inputs: none.
        Output: True when the Grouping controls are visible.
        Uses: saving the review window settings.
        Limitations: this is the header state. It does not say what the burst gap is.
        """
        return self._grouping_toggle.isChecked()

    def scoring_open(self) -> bool:
        """
        Read whether the Scoring section is open.

        Inputs: none.
        Output: True when the Scoring controls are visible.
        Uses: saving the review window settings.
        Limitations: this is the header state. It does not say whether subject scoring is on.
        """
        return self._scoring_toggle.isChecked()

    def set_grouping_open(self, open_: bool) -> None:
        """
        Open or close Grouping without treating it as a user click.

        Inputs: open_, whether the Grouping controls are visible.
        Output: no return value. The header and the body match open_.
        Uses: restoring the saved Grouping section.
        Limitations: the signal is blocked, so restoring does not emit a second change. The burst gap is left as it is.
        """
        self._grouping_toggle.blockSignals(True)
        self._grouping_toggle.setChecked(bool(open_))
        self._grouping_toggle.blockSignals(False)
        self._apply_section(self._grouping_toggle, self._grouping_body, bool(open_))

    def set_scoring_open(self, open_: bool) -> None:
        """
        Open or close Scoring without treating it as a user click.

        Inputs: open_, whether the Scoring controls are visible.
        Output: no return value. The header and the body match open_.
        Uses: restoring the saved Scoring section.
        Limitations: the signal is blocked, so restoring does not emit a second change. The sliders are left as they are.
        """
        self._scoring_toggle.blockSignals(True)
        self._scoring_toggle.setChecked(bool(open_))
        self._scoring_toggle.blockSignals(False)
        self._apply_section(self._scoring_toggle, self._scoring_body, bool(open_))

    def set_minimum_score_enabled(self, checked: bool) -> None:
        """
        Set the minimum-score checkbox without emitting a change.

        Inputs: checked, whether frames below the cutoff are hidden.
        Output: no return value. The checkbox matches checked, and the spin box is enabled only when checked is true.
        Uses: restoring the saved Minimum score setting.
        Limitations: signals are blocked, so the burst list is not refreshed by this call.
        """
        self._score_filter.blockSignals(True)
        self._score_filter.setChecked(bool(checked))
        self._score_minimum.setEnabled(bool(checked))
        self._score_filter.blockSignals(False)

    def set_minimum_score(self, value: int) -> None:
        """
        Set the minimum score without emitting a change.

        Inputs: value, a score from 0 to 100. Values outside that range are clamped.
        Output: no return value. The spin box shows the clamped value.
        Uses: restoring the saved Minimum score setting.
        Limitations: signals are blocked, so the burst list is not refreshed by this call. This does not turn the filter on.
        """
        self._score_minimum.blockSignals(True)
        self._score_minimum.setValue(max(0, min(100, int(value))))
        self._score_minimum.blockSignals(False)

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
        Read whether bird eye detection is requested.

        Inputs: none.
        Output: True when Bird eye detection is checked.
        Uses: the Scoring section and the scan worker.
        Limitations: the checkbox does not rescan by itself. The review window downloads the weights when they are missing, then rescans.
        """
        return self._eye.isChecked()

    def show_subject_box(self) -> bool:
        """
        Read whether the detected animal box is drawn.

        Inputs: none.
        Output: True when Show subject box is checked.
        Uses: the preview and the compare pane.
        Limitations: this does not say whether a box was found. Subject scoring is a different checkbox.
        """
        return self._show_box.isChecked()

    def show_eye_circle(self) -> bool:
        """
        Read whether the bird eye ring is drawn.

        Inputs: none.
        Output: True when Show eye circle is checked.
        Uses: the preview and the compare pane.
        Limitations: this does not say whether an eye was found. Bird eye detection is a different checkbox.
        """
        return self._show_eye.isChecked()

    def weights(self) -> Weights:
        """
        Build the scoring weights from the sliders and the subject checkbox.

        Inputs: none.
        Output: a Weights value. Sharpness and center are fractions of the slider percents. Exposure is the share left after sharpness, and never below zero.
        Uses: rescoring an open library and the next scan.
        Limitations: bird eye detection is a separate checkbox and is not part of the returned Weights.
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
        Set the bird-eye checkbox without emitting a change.

        Inputs: checked, whether Bird eye detection is on.
        Output: no return value. The checkbox matches checked.
        Uses: restoring the saved setting, and turning it off when the weights download fails or is cancelled.
        Limitations: signals are blocked, so this call does not rescan or download.
        """
        self._eye.blockSignals(True)
        self._eye.setChecked(checked)
        self._eye.blockSignals(False)

    def detection_installed(self) -> bool:
        """
        Report whether the subject and bird eye controls are on screen.

        Inputs: none.
        Output: True when those controls are visible.
        Uses: the review window after the modules are installed.
        Limitations: this follows the visibility set by set_detection_installed. A collapsed Scoring section can hide them even when this returns True.
        """
        return not self._detection.isHidden()

    def set_detection_installed(self, installed: bool) -> None:
        """
        Show the detection controls, or the install button.

        Inputs: installed, whether subject and bird eye detection are already installed.
        Output: no return value. The controls are shown when installed is True. The install button is shown when it is False.
        Uses: opening the window, and the end of a successful install.
        Limitations: this does not install software and does not change the checkbox values.
        """
        self._detection.setVisible(installed)
        self._install_detection.setVisible(not installed)

    def set_show_subject_box(self, checked: bool) -> None:
        """
        Set the subject-box checkbox without emitting a change.

        Inputs: checked, whether the detected animal box is drawn.
        Output: no return value. The checkbox matches checked.
        Uses: restoring the saved Show subject box setting.
        Limitations: signals are blocked, so the preview is not redrawn by this call. Subject scoring is left as it is.
        """
        self._show_box.blockSignals(True)
        self._show_box.setChecked(bool(checked))
        self._show_box.blockSignals(False)

    def set_show_eye_circle(self, checked: bool) -> None:
        """
        Set the eye-circle checkbox without emitting a change.

        Inputs: checked, whether the bird eye ring is drawn.
        Output: no return value. The checkbox matches checked.
        Uses: restoring the saved Show eye circle setting.
        Limitations: signals are blocked, so the preview is not redrawn by this call. Bird eye detection is left as it is.
        """
        self._show_eye.blockSignals(True)
        self._show_eye.setChecked(bool(checked))
        self._show_eye.blockSignals(False)

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
