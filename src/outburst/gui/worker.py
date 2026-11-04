"""Background scan so the window stays responsive while previews are scored."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from outburst.models import Weights
from outburst.pipeline import CancelledError, analyze_folder

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
    ) -> None:
        super().__init__()
        self._folder = folder
        self._recursive = recursive
        self._threshold = threshold
        self._weights = weights
        self._cache_dir = cache_dir
        self._subject_model = subject_model
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
