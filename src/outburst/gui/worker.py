"""Background threads so the window stays responsive.

``AnalyzeWorker`` scans a folder and can be aborted. It can pass a bird-eye weights path. ``ModelFetchWorker`` downloads one weights file, including the bird-eye pose model. ``RuntimeInstallWorker`` installs Ultralytics when the photographer uses Install subject and bird eye detection. ``ExportWorker`` copies keepers two at a time and writes an ``.xmp`` beside each saved file. ``SidecarWorker`` writes XMP for keepers and star-rated frames. Each ``run`` emits progress, then success or failure, on the Qt thread.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from outburst.models import Library, Weights
from outburst.output import export_keepers, write_exported_sidecars, write_library_sidecars
from outburst.pipeline import CancelledError, analyze_folder
from outburst.subject import (
    ModelCancelled,
    SubjectRuntimeCancelled,
    ensure_model,
    install_subject_runtime,
    option_by_id,
    subject_runtime_ready,
)

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
        eye_model: Path | None = None,
    ) -> None:
        """
        Store the folder and scoring options for a background scan.

        Inputs: folder; recursive; threshold, the burst gap in seconds; weights; cache_dir; subject_model, or None; eye_scoring; eye_model, the bird-eye weights, or None.
        Output: no return value. The worker is ready and its abort flag is clear.
        Uses: the scan that fills the review window.
        Limitations: storing the options does not start the scan. The thread runs when start is called.
        """
        super().__init__()
        self._folder = folder
        self._recursive = recursive
        self._threshold = threshold
        self._weights = weights
        self._cache_dir = cache_dir
        self._subject_model = subject_model
        self._eye_scoring = eye_scoring
        self._eye_model = eye_model
        self._abort = False

    def abort(self) -> None:
        """
        Ask the running scan to stop.

        Inputs: none.
        Output: no return value. The abort flag is set for the pipeline's cancel check.
        Uses: Cancel scan and closing the review window.
        Limitations: the scan stops when the pipeline next checks the flag, so the current step can finish.
        """
        self._abort = True

    def run(self) -> None:
        """
        Scan the folder and emit the finished library, a failure, or cancellation.

        Inputs: none. The folder and options were stored on the worker.
        Output: no return value. Emits succeeded with a library, failed with the exception text, or cancelled.
        Uses: the background scan behind the review window.
        Limitations: an abort discovered after a successful scan emits cancelled and does not emit the library. Original files are not modified.
        """
        try:
            library = analyze_folder(
                self._folder,
                recursive=self._recursive,
                threshold=self._threshold,
                weights=self._weights,
                cache_dir=self._cache_dir,
                subject_model=self._subject_model,
                eye_scoring=self._eye_scoring,
                eye_model=self._eye_model,
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
        """
        Forward one scan progress update to the window.

        Inputs: done, the finished count; total, the expected count; message, the status text.
        Output: no return value. Emits the progress signal with those three values.
        Uses: the status bar and progress bar during a scan.
        Limitations: this method emits the signal and does not update widgets itself.
        """
        self.progress.emit(done, total, message)


class ModelFetchWorker(QThread):
    """Download one subject model without blocking the window."""

    progress = Signal(int, int)
    succeeded = Signal(str)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(self, model_id: str, directory: Path) -> None:
        """
        Store which subject model to download and where to save it.

        Inputs: model_id, a subject-model identifier; directory, the models folder.
        Output: no return value. The worker is ready and its abort flag is clear.
        Uses: subject scoring setup, and the bird-eye download, before a scan.
        Limitations: an unknown model id is reported later, when run executes. The bird-eye id is eye-pose-v0.
        """
        super().__init__()
        self._model_id = model_id
        self._directory = directory
        self._abort = False

    def abort(self) -> None:
        """
        Ask the model download to stop.

        Inputs: none.
        Output: no return value. The abort flag is set for the downloader's cancel check.
        Uses: Cancel download and closing the review window.
        Limitations: the download stops when the downloader next checks the flag.
        """
        self._abort = True

    def run(self) -> None:
        """
        Download the selected subject model and emit its saved path.

        Inputs: none. The model id and directory were stored on the worker.
        Output: no return value. Emits succeeded with the path, failed with a message, or cancelled.
        Uses: the subject-model download before a scan.
        Limitations: an unknown model id emits failed with "Unknown subject model" and does not write a file. An abort after a successful download emits cancelled.
        """
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


class RuntimeInstallWorker(QThread):
    """Install Ultralytics without blocking the window."""

    succeeded = Signal()
    failed = Signal(str)
    cancelled = Signal()
    status = Signal(str)

    def __init__(self) -> None:
        """
        Prepare a background install of the subject extra.

        Inputs: none.
        Output: no return value. The worker is ready and its abort flag is clear.
        Uses: subject detection and bird eye detection when Ultralytics is not imported yet.
        Limitations: constructing the worker does not start pip.
        """
        super().__init__()
        self._abort = False

    def abort(self) -> None:
        """
        Ask the Ultralytics install to stop.

        Inputs: none.
        Output: no return value. The abort flag is set for the installer's cancel check.
        Uses: Cancel download and closing the review window during the install.
        Limitations: pip stops the next time the installer checks the flag, then the process is terminated.
        """
        self._abort = True

    def run(self) -> None:
        """
        Install Ultralytics and emit success, failure, or cancellation.

        Inputs: none.
        Output: no return value. Emits succeeded when the import works, failed with the pip error, or cancelled. Emits status with the current install step.
        Uses: the review window before a scan that needs subject or bird eye detection.
        Limitations: the download includes PyTorch and can take several minutes. Model weights are not downloaded here. An abort after a successful install emits cancelled and does not emit success.
        """
        try:
            install_subject_runtime(
                should_cancel=lambda: self._abort,
                on_status=self.status.emit,
            )
        except SubjectRuntimeCancelled:
            self.cancelled.emit()
            return
        except Exception as exc:
            log.exception("Subject runtime install failed")
            self.failed.emit(str(exc))
            return
        if self._abort:
            self.cancelled.emit()
            return
        if not subject_runtime_ready():
            self.failed.emit(
                "Ultralytics installed, but this window still cannot import it. "
                "Quit Outburst and open it again."
            )
            return
        self.succeeded.emit()


class ExportWorker(QThread):
    """Copy or link keepers without blocking the window, then write one sidecar each."""

    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        sources: list[Path],
        folder: Path,
        mode: str,
        library: Library,
        decisions: dict[Path, str],
        stars: dict[Path, int],
    ) -> None:
        """
        Store the keeper files, destination folder, copy or link mode, and the marks for the sidecars.

        Inputs: sources, the files to save; folder, the destination; mode, "copy" or "hardlink"; library; decisions, each path's keep or reject mark; stars, each path's rating.
        Output: no return value. The worker is ready to start.
        Uses: Save keepers.
        Limitations: this worker has no abort flag. Original files are not modified. It keeps the decision and star maps it was given.
        """
        super().__init__()
        self._sources = sources
        self._folder = folder
        self._mode = mode
        self._library = library
        self._decisions = decisions
        self._stars = stars

    def run(self) -> None:
        """
        Copy or link the keeper files, write an .xmp beside each one, and emit the results.

        Inputs: none. Sources, folder, mode, library, and marks were stored on the worker.
        Output: no return value. Emits succeeded with the export results, or failed with the exception text.
        Uses: Save keepers, off the window thread.
        Limitations: the copies and the sidecars each run two at a time. A file already in the destination is left as it is. The sidecar is written in the keepers folder. Original files are not modified.
        """
        try:
            count = len(self._sources)

            def report_copy(done: int, total: int, message: str) -> None:
                """Count each saved photograph as the first half of the save.

                Inputs: done, how many photographs are saved; total, the photograph count; message, the status line.
                Output: none. Progress is emitted with a total that leaves room for the sidecars.
                Uses: export_keepers during Save keepers.
                Limitations: an empty save reports a total of 2. This does not write a file.
                """

                self.progress.emit(done, max(total, count, 1) * 2, message)

            def report_sidecar(done: int, total: int, message: str) -> None:
                """Count each sidecar after the photographs.

                Inputs: done, how many sidecars are written; total, the sidecar count; message, the status line.
                Output: none. Progress continues from the end of the photograph copies.
                Uses: write_exported_sidecars during Save keepers.
                Limitations: the total matches the copy phase. This does not write a file.
                """

                self.progress.emit(max(total, count) + done, max(total, count, 1) * 2, message)

            results = export_keepers(
                self._sources,
                self._folder,
                self._mode,
                progress=report_copy,
            )
            results, _warnings = write_exported_sidecars(
                self._library,
                results,
                self._decisions,
                stars=self._stars,
                progress=report_sidecar,
            )
        except Exception as exc:
            log.exception("Saving keepers failed")
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(results)


class SidecarWorker(QThread):
    """Write XMP sidecars without blocking the window. Two files at a time."""

    progress = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        library: Library,
        decisions: dict[Path, str],
        stars: dict[Path, int],
        minimum_frames: int,
    ) -> None:
        """
        Store the library and the marks to write into sidecars.

        Inputs: library; decisions, each path's keep or reject mark; stars, each path's rating; minimum_frames.
        Output: no return value. The worker is ready to start.
        Uses: Write XMP sidecars.
        Limitations: this worker has no abort flag. It keeps the decision and star maps it was given.
        """
        super().__init__()
        self._library = library
        self._decisions = decisions
        self._stars = stars
        self._minimum_frames = minimum_frames

    def run(self) -> None:
        """
        Write XMP sidecars for the marked frames and emit what was written.

        Inputs: none. The library, decisions, stars, and minimum were stored on the worker.
        Output: no return value. Emits succeeded with the written paths and warnings, or failed with the exception text.
        Uses: Write XMP sidecars, off the window thread.
        Limitations: sidecars are keepers and star-rated frames only. The writes run two at a time. Original files are not modified.
        """
        try:
            written, warnings = write_library_sidecars(
                self._library,
                self._decisions,
                stars=self._stars,
                minimum_frames=self._minimum_frames,
                progress=lambda done, total, message: self.progress.emit(done, total, message),
            )
        except Exception as exc:
            log.exception("Writing sidecars failed")
            self.failed.emit(str(exc))
            return
        self.succeeded.emit((written, warnings))
