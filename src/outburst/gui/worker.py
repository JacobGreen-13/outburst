"""Background scan so the window stays responsive while previews are scored."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from outburst.models import Weights
from outburst.pipeline import CancelledError, analyze_folder
from outburst.subject import ModelCancelled, ensure_model, option_by_id

log = logging.getLogger(__name__)


class AnalyzeWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        folder: Path,
        *,
        recursive: bool,
        threshold: float,
        weights: Weights,
        cache_dir: Path,
        subject_model: Path | None,
        eye_scoring: bool = False,
    ) -> None:
        super().__init__()
        self._folder = folder
        self._recursive = recursive
        self._threshold = threshold
        self._weights = weights
        self._cache_dir = cache_dir
        self._subject_model = subject_model
        self._eye_scoring = eye_scoring
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        try:
            library = analyze_folder(
                self._folder,
                recursive=self._recursive,
                threshold=self._threshold,
                weights=self._weights,
                cache_dir=self._cache_dir,
                subject_model=self._subject_model,
                eye_scoring=self._eye_scoring,
                progress=self._on_progress,
                should_cancel=lambda: self._abort,
            )
        except CancelledError:
            self.cancelled.emit()
            return
        except Exception as exc:
            log.exception("Scan failed")
            self.failed.emit(str(exc))
            return
        if self._abort:
            self.cancelled.emit()
            return
        self.succeeded.emit(library)

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.emit(done, total, message)


class ModelFetchWorker(QThread):
    """Download one subject model without blocking the window."""

    progress = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, model_id: str, directory: Path) -> None:
        super().__init__()
        self._model_id = model_id
        self._directory = directory
        self._abort = False

    def abort(self) -> None:
        self._abort = True

    def run(self) -> None:
        option = option_by_id(self._model_id)
        if option is None:
            self.failed.emit("Unknown subject model")
            return
        try:
            path = ensure_model(
                option,
                self._directory,
                progress=lambda done, total: self.progress.emit(done, total or 0),
                should_cancel=lambda: self._abort,
            )
        except ModelCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            log.exception("Model download failed")
            self.failed.emit(str(exc))
            return
        if self._abort:
            self.cancelled.emit()
            return
        self.succeeded.emit(str(path))
