"""Scan a folder, score previews, and group bursts.

This is the part the window calls. It reads photographs and writes only inside
the cache directory. Original files are opened for reading.

``analyze_folder`` lists, scores, and groups a folder. ``with_weights`` recomputes composites from cached measurements. ``with_threshold`` regroups without rereading files. ``_score_all`` and ``_measure_with_detector`` run the worker pools. The other helpers decode one preview, detect a subject batch, read bird eyes on the coordinating thread, score the preview, and store the cache row.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from outburst.cache import Cache, default_cache_dir
from outburst.eyes import EYE_CACHE_TOKEN, load_bird_eye_detector
from outburst.grouping import group_frames
from outburst.metadata import read_capture_info, timestamp_reader_warning
from outburst.models import Frame, FrameAnalysis, Library, Weights
from outburst.parallel import SUBJECT_BATCH, preview_workers, run_ordered
from outburst.preview import (
    PreviewError,
    decode_image,
    encode_jpeg,
    extract_preview_bytes,
    resize_max_edge,
)
from outburst.scanner import ScannedFile, scan_folder
from outburst.scoring import composite_score, score_bgr
from outburst.subject import SubjectDetector, load_yolo_detector, make_subject_key

log = logging.getLogger(__name__)

ANALYSIS_EDGE = 1024
THUMB_EDGE = 360

ProgressCallback = Callable[[int, int, str], None]


class CancelledError(RuntimeError):
    """The photographer cancelled the scan."""


def analyze_folder(
    folder: Path,
    *,
    recursive: bool = False,
    threshold: float = 1.0,
    weights: Weights | None = None,
    cache_dir: Path | None = None,
    subject_model: Path | None = None,
    detector: SubjectDetector | None = None,
    eye_scoring: bool = False,
    eye_model: Path | None = None,
    eye_finder: object | None = None,
    progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Library:
    """Score every supported photograph in `folder` and group them into bursts.

    Inputs: folder, the directory to scan; recursive, whether subfolders are included; threshold, the burst gap in seconds; weights, the sharpness blend; cache_dir, an optional cache location; subject_model, an optional weights path; detector, an optional already-loaded detector; eye_scoring, whether bird eyes are measured; eye_model, the pose weights to load when eye_scoring is on; eye_finder, an optional already-loaded finder; progress, an optional count callback; should_cancel, an optional cancel check.
    Output: a Library of analyses, bursts, and warnings.
    Uses: the window's scan.
    Limitations: photographs are opened for reading only. Previews are an embedded JPEG or a paired JPEG, with the long edge resized to 1024 px, and there is no demosaic. The subject model and the bird-eye model stay on this coordinating thread. Cache rows are written on this thread. The bird-eye cache key includes a generation token, so a scan from before that token is scored again. Consecutive frames share a burst only while the gap is strictly less than threshold. Cancel raises CancelledError. eye_scoring with no weights and no finder leaves bird eyes off and adds a warning.
    """

    weights = (weights or Weights()).clamped()
    folder = folder.expanduser().resolve()
    warnings: list[str] = []
    subject_active = False
    subject_key = ""
    if detector is not None:
        subject_active = True
        subject_key = "custom-detector"
    elif subject_model is not None:
        try:
            detector = load_yolo_detector(subject_model)
            subject_key = make_subject_key(subject_model)
            subject_active = True
        except Exception as exc:
            warnings.append(f"Subject detection stayed off: {exc}")
            detector = None
            subject_key = ""
    eye_active = False
    if eye_scoring:
        if eye_finder is not None:
            eye_active = True
            subject_key = f"{subject_key}|bird-eye" if subject_key else "bird-eye"
        elif eye_model is not None:
            try:
                eye_finder = load_bird_eye_detector(eye_model)
                eye_key = f"{make_subject_key(eye_model)}|{EYE_CACHE_TOKEN}"
                subject_key = f"{subject_key}|{eye_key}" if subject_key else eye_key
                eye_active = True
            except Exception as exc:
                warnings.append(f"Bird eye detection stayed off: {exc}")
                eye_finder = None
        else:
            warnings.append("Bird eye detection stayed off: the weights are not in the models folder.")
            eye_finder = None

    files = scan_folder(folder, recursive=recursive)
    log.info("Found %d file(s) in %s", len(files), folder)
    if not files:
        _report(progress, 0, 0, "No photographs found")
        return Library(
            folder,
            (),
            (),
            ("No RAW or JPEG files were found in that folder.",),
            threshold,
            weights,
            False,
            eye_active,
        )

    reader_note = timestamp_reader_warning(item.path.suffix for item in files if item.kind == "raw")
    if reader_note:
        warnings.append(reader_note)

    cache: Cache | None
    try:
        cache = Cache(cache_dir or default_cache_dir())
    except OSError as exc:
        log.warning("Score cache is unavailable: %s", exc)
        warnings.append(f"Score cache is unavailable ({exc}). This scan will be slower next time too.")
        cache = None

    try:
        analyses, detection_failures = _score_all(
            files,
            cache=cache,
            weights=weights,
            detector=detector,
            subject_key=subject_key,
            eye_finder=eye_finder,
            progress=progress,
            should_cancel=should_cancel,
        )
    finally:
        if cache is not None:
            cache.close()

    if detection_failures:
        warnings.append(
            f"Subject detection failed on {detection_failures} file(s). "
            "Those frames use full-frame sharpness."
        )
    missing = [item for item in analyses if item.frame.captured_at is None]
    if missing:
        preview = ", ".join(item.frame.path.name for item in missing[:5])
        extra = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        warnings.append(
            f"{len(missing)} file(s) had no capture time and were left as their own bursts: {preview}{extra}"
        )
    failed = [item for item in analyses if item.error]
    if failed:
        preview = ", ".join(item.frame.path.name for item in failed[:5])
        extra = f" (+{len(failed) - 5} more)" if len(failed) > 5 else ""
        warnings.append(f"{len(failed)} file(s) had no usable preview: {preview}{extra}")

    bursts = tuple(group_frames([item.frame for item in analyses], threshold))
    log.info("Grouped %d file(s) into %d burst(s)", len(analyses), len(bursts))
    return Library(
        folder=folder,
        analyses=tuple(analyses),
        bursts=bursts,
        warnings=tuple(warnings),
        threshold=threshold,
        weights=weights,
        subject_active=subject_active,
        eye_active=eye_active,
    )


def with_weights(library: Library, weights: Weights) -> Library:
    """Recompute composite scores from cached measurements.

    Inputs: library, an existing scan; weights, the new sharpness, exposure, and center mix.
    Output: a library with the same frames and bursts and new composite scores.
    Uses: the sharpness, exposure, and center sliders.
    Limitations: measurements are not reread from the photographs. A frame with no metrics gets a None composite.
    """

    weights = weights.clamped()
    analyses = tuple(_retarget(item, weights) for item in library.analyses)
    return replace(library, analyses=analyses, weights=weights)


def with_threshold(library: Library, threshold: float) -> Library:
    """Regroup the same frames under a new gap threshold.

    Inputs: library, an existing scan; threshold, the new gap in seconds.
    Output: a library with the same analyses and new bursts.
    Uses: the burst-gap control.
    Limitations: frames are not reread. Consecutive frames stay together only while the gap is strictly less than the threshold. A frame with no capture time stays in its own burst.
    """

    bursts = tuple(group_frames([item.frame for item in library.analyses], threshold))
    return replace(library, bursts=bursts, threshold=float(threshold))


def _score_all(
    files: list[ScannedFile],
    *,
    cache: Cache | None,
    weights: Weights,
    detector: SubjectDetector | None,
    subject_key: str,
    eye_finder: object | None,
    progress: ProgressCallback | None,
    should_cancel: Callable[[], bool] | None,
) -> tuple[list[FrameAnalysis], int]:
    """Reuse cache hits and measure the rest, preserving folder order.

    Inputs: files, the scanned photographs; cache, the score cache or None; weights, the current blend; detector, an optional subject detector; subject_key, the cache identity for the models; eye_finder, an optional bird-eye model; progress and should_cancel, optional callbacks.
    Output: analyses in file order, and the count of frames whose subject detection failed.
    Uses: analyze_folder.
    Limitations: cache rows are read here. New cache rows are written later on the coordinating thread, not on the preview workers. Photographs are not opened for writing. The bird-eye model is not run on the preview workers.
    """

    cached_rows: dict[Path, FrameAnalysis] = {}
    pending: list[ScannedFile] = []
    total = len(files)
    seen = 0
    for scanned in files:
        _check_cancel(should_cancel)
        cached = _from_cache(cache, scanned, weights, subject_key)
        if cached is not None:
            cached_rows[scanned.path] = cached
            seen += 1
            _report(progress, seen, total, f"Cached {scanned.path.name}")
        else:
            pending.append(scanned)

    measured: dict[Path, tuple[FrameAnalysis, int]] = {}
    if pending:
        _report(progress, seen, total, "Reading capture times")
        capture_info = read_capture_info([item.path for item in pending])
        measured = _measure_pending(
            pending,
            capture_info=capture_info,
            cache=cache,
            weights=weights,
            detector=detector,
            subject_key=subject_key,
            eye_finder=eye_finder,
            progress=progress,
            done_offset=seen,
            total=total,
            should_cancel=should_cancel,
        )

    detection_failures = 0
    analyses: list[FrameAnalysis] = []
    for scanned in files:
        cached = cached_rows.get(scanned.path)
        if cached is not None:
            analyses.append(cached)
            continue
        analysis, failed = measured[scanned.path]
        detection_failures += failed
        analyses.append(analysis)
    return analyses, detection_failures


def _measure_pending(
    pending: list[ScannedFile],
    *,
    capture_info: dict,
    cache: Cache | None,
    weights: Weights,
    detector: SubjectDetector | None,
    subject_key: str,
    eye_finder: object | None,
    progress: ProgressCallback | None,
    done_offset: int,
    total: int,
    should_cancel: Callable[[], bool] | None,
) -> dict[Path, tuple[FrameAnalysis, int]]:
    """Score cache misses. Preview work uses a small pool. The cache is written here.

    Inputs: pending, the cache misses; capture_info, clocks keyed by path; cache, weights, detector, subject_key, and eye_finder, the measurement settings; progress, done_offset, and total, the status line; should_cancel, an optional cancel check.
    Output: a dict from path to the analysis and a detection-failure count of 0 or 1.
    Uses: _score_all for files that were not in the cache.
    Limitations: without a detector or a bird-eye finder, workers score previews and persist runs on this thread. With either model, that model stays on this thread. Cache rows are written in persist, not on the workers. Originals are not written.
    """

    found: dict[Path, tuple[FrameAnalysis, int]] = {}

    def persist(index: int, pair: tuple[FrameAnalysis, int]) -> None:
        """Store one finished analysis in the cache and in the result map.

        Inputs: index, the position in pending; pair, the analysis and its detection-failure count.
        Output: none. The cache row is written and the pair is stored under the frame path.
        Uses: _measure_pending. The ordered pool and the detector loop call this on the coordinating thread.
        Limitations: cache writes happen here, not on a worker thread. A missing cache, or an analysis with no metrics or preview files, is not stored. The photograph is not written.
        """

        analysis, _failed_count = pair
        _store_analysis(cache, analysis, subject_key)
        found[analysis.frame.path] = pair
        _report(progress, done_offset + index + 1, total, f"Scored {pending[index].path.name}")

    def cancel() -> None:
        """Stop the measurement when the photographer has cancelled.

        Inputs: none. The enclosing should_cancel callback is read.
        Output: none. CancelledError is raised when the callback returns true.
        Uses: the preview pool and the detector loop between batches.
        Limitations: the check runs on the coordinating thread. It does not roll back cache rows already written.
        """

        _check_cancel(should_cancel)

    if detector is None and eye_finder is None:

        def score_at(index: int) -> tuple[FrameAnalysis, int]:
            """Load and score one cache miss when no model is running.

            Inputs: index, the position in pending.
            Output: the analysis and a detection-failure count of 0.
            Uses: _measure_pending. A preview worker calls this.
            Limitations: no subject model and no bird-eye model run on this call. The preview is an embedded or paired JPEG resized to a 1024 px long edge, with no demosaic. The SQLite cache row is written later on the coordinating thread.
            """

            scanned = pending[index]
            return _score_loaded(
                scanned,
                capture_info.get(scanned.path),
                cache,
                weights,
                subject_key,
                None,
                0,
                None,
            )

        run_ordered(
            len(pending),
            score_at,
            workers=preview_workers(len(pending)),
            cancel=cancel,
            on_finished=persist,
        )
        return found

    _measure_with_detector(
        pending,
        capture_info=capture_info,
        cache=cache,
        weights=weights,
        detector=detector,
        subject_key=subject_key,
        eye_finder=eye_finder,
        persist=persist,
        cancel=cancel,
    )
    return found


@dataclass
class _Loaded:
    scanned: ScannedFile
    frame: Frame
    image: np.ndarray | None
    error: str | None


def _measure_with_detector(
    pending: list[ScannedFile],
    *,
    capture_info: dict,
    cache: Cache | None,
    weights: Weights,
    detector: SubjectDetector | None,
    subject_key: str,
    eye_finder: object | None,
    persist: Callable[[int, tuple[FrameAnalysis, int]], None],
    cancel: Callable[[], None],
) -> None:
    """Decode ahead on a pool, run models on this thread, then score.

    A model is not shared across threads. A batch is a few frames. Decode of
    the next frames keeps running while that batch is predicted.

    Inputs: pending, the cache misses; capture_info, clocks; cache, weights, detector, subject_key, and eye_finder, the measurement settings; persist, the coordinating-thread callback that stores one result; cancel, the cancel check.
    Output: none. Results are delivered through persist in file order.
    Uses: _measure_pending when a subject detector or a bird-eye finder is active.
    Limitations: detect_many and find_many run on this caller thread, in batches of up to four frames. The models are not shared across threads. Cache rows stay in persist. Photographs are not written, and previews are not demosaiced.
    """

    total = len(pending)
    workers = preview_workers(total)
    decoded: dict[int, _Loaded] = {}
    measured: dict[int, tuple[FrameAnalysis, int]] = {}
    next_decode = 0
    next_detect = 0
    next_persist = 0
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="outburst-preview")
    decode_flight: dict[Future[_Loaded], int] = {}
    measure_flight: dict[Future[tuple[FrameAnalysis, int]], int] = {}

    def fill_decodes() -> int:
        """Submit preview decodes until the worker cap or the look-ahead cap is reached.

        Inputs: none. The enclosing pool, pending list, and cursors are used.
        Output: how many decode tasks were submitted on this call.
        Uses: _measure_with_detector so the next frames decode while a batch is predicted.
        Limitations: at most the preview-worker count is in flight, and look-ahead stays within that count plus SUBJECT_BATCH of the last persisted frame. Decodes do not write the cache and do not run the model.
        """

        nonlocal next_decode
        submitted = 0
        while (
            next_decode < total
            and len(decode_flight) < workers
            and next_decode - next_persist < workers + SUBJECT_BATCH
        ):
            index = next_decode
            info = capture_info.get(pending[index].path)
            decode_flight[pool.submit(_load_scanned, pending[index], info)] = index
            next_decode += 1
            submitted += 1
        return submitted

    try:
        while next_persist < total:
            cancel()
            _collect(decode_flight, decoded)
            _collect(measure_flight, measured)
            fill_decodes()
            progressed = False
            force = next_decode == total and not decode_flight
            group = _next_detect_group(decoded, next_detect, total, force)
            if group is not None and group[0] == "error":
                loaded = decoded[group[1][0]]
                measured[group[1][0]] = (_failed(loaded.frame, loaded.error or "No preview"), 0)
                next_detect = group[1][0] + 1
                progressed = True
            elif group is not None and len(measure_flight) < workers:
                fill_decodes()
                indexes = group[1]
                images: list[np.ndarray] = []
                for index in indexes:
                    image = decoded[index].image
                    if image is None:
                        raise RuntimeError("A frame in the subject batch has no preview")
                    images.append(image)
                if detector is None:
                    boxes: list[object] = [None] * len(images)
                    failures = [0] * len(images)
                else:
                    boxes, failures = _detect_batch(detector, images)
                eyes = _find_eyes(eye_finder, images)
                for offset, index in enumerate(indexes):
                    loaded = decoded[index]
                    image = images[offset]
                    loaded.image = None
                    measure_flight[pool.submit(
                        _score_image,
                        loaded,
                        image,
                        boxes[offset],
                        failures[offset],
                        eyes[offset],
                        cache,
                        weights,
                        subject_key,
                    )] = index
                next_detect = indexes[-1] + 1
                progressed = True
            while next_persist in measured:
                persist(next_persist, measured.pop(next_persist))
                decoded.pop(next_persist, None)
                next_persist += 1
                progressed = True
            if next_persist >= total:
                break
            if progressed:
                continue
            waiting = [*decode_flight.keys(), *measure_flight.keys()]
            if not waiting:
                raise RuntimeError("Scoring stalled before every frame was measured")
            wait(waiting, timeout=0.25, return_when=FIRST_COMPLETED)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def _next_detect_group(
    decoded: dict[int, _Loaded],
    start: int,
    total: int,
    force: bool,
) -> tuple[str, list[int]] | None:
    """Take the next ready frames for one subject batch, or a single decode failure.

    Inputs: decoded, loaded frames by index; start, the next index to detect; total, the number of pending files; force, whether every decode has already finished.
    Output: an error pair for one index when that frame has no preview, a batch pair when a full batch is ready or force drains a short tail, or None when the next frame is not ready.
    Uses: _measure_with_detector.
    Limitations: a batch stops at SUBJECT_BATCH frames or at the first frame that failed to decode. A partial batch waits unless force is set.
    """

    if start not in decoded:
        return None
    if decoded[start].image is None:
        return ("error", [start])
    indexes: list[int] = []
    cursor = start
    while cursor < total and cursor in decoded and len(indexes) < SUBJECT_BATCH:
        if decoded[cursor].image is None:
            break
        indexes.append(cursor)
        cursor += 1
    if len(indexes) >= SUBJECT_BATCH or (force and indexes):
        return ("batch", indexes)
    return None


def _collect(in_flight: dict[Future, int], destination: dict) -> None:
    """Move finished futures into a result dict without waiting.

    Inputs: in_flight, futures mapped to their indexes; destination, the dict that receives results.
    Output: none. Completed futures are popped and their results stored by index.
    Uses: the detector loop for both decode and score futures.
    Limitations: the wait timeout is zero, so only futures that are already done are collected. A failed future raises when its result is read.
    """

    if not in_flight:
        return
    done, _pending = wait(set(in_flight), timeout=0, return_when=FIRST_COMPLETED)
    for future in done:
        index = in_flight.pop(future)
        destination[index] = future.result()


def _detect_batch(
    detector: SubjectDetector,
    images: list[np.ndarray | None],
) -> tuple[list[object], list[int]]:
    """Run subject detection for one batch on the caller thread.

    Inputs: detector, the subject model; images, the BGR previews. A None entry is rejected.
    Output: one box or None per image, and a matching list of 0 or 1 failure flags.
    Uses: _measure_with_detector.
    Limitations: detect_many is preferred and runs on this thread. The model is not shared across threads. A detection exception marks every frame in that batch as failed and returns no boxes. Those frames then use full-frame sharpness.
    """

    ready = [image for image in images if image is not None]
    if len(ready) != len(images):
        raise RuntimeError("A frame in the subject batch has no preview")
    many = getattr(detector, "detect_many", None)
    if callable(many):
        try:
            found = list(many(ready))
        except Exception:
            log.warning("Subject detection failed for %d frame(s)", len(ready), exc_info=True)
            return [None] * len(ready), [1] * len(ready)
        if len(found) != len(ready):
            log.warning("Subject detection returned %d boxes for %d frames", len(found), len(ready))
            found = (found + [None] * len(ready))[: len(ready)]
            return found, [1] * len(ready)
        return found, [0] * len(ready)
    found: list[object] = []
    failures: list[int] = []
    for image in ready:
        try:
            found.append(detector.detect(image))
            failures.append(0)
        except Exception:
            log.warning("Subject detection failed", exc_info=True)
            found.append(None)
            failures.append(1)
    return found, failures


def _find_eyes(finder: object | None, images: list[np.ndarray]) -> list[object]:
    """Run bird-eye detection for one batch on the caller thread.

    Inputs: finder, the pose model or None; images, the BGR previews.
    Output: one eye or None per image. A missing finder returns None for each image.
    Uses: _measure_with_detector.
    Limitations: find_many runs on this thread. The model is not shared across threads. A detection exception returns no eyes for that batch. Those frames keep their usual sharpness rank.
    """

    if finder is None:
        return [None] * len(images)
    many = getattr(finder, "find_many", None)
    if not callable(many):
        log.warning("Bird eye finder has no find_many method")
        return [None] * len(images)
    try:
        found = list(many(images))
    except Exception:
        log.warning("Bird eye detection failed for %d frame(s)", len(images), exc_info=True)
        return [None] * len(images)
    if len(found) != len(images):
        log.warning("Bird eye detection returned %d eyes for %d frames", len(found), len(images))
        found = (found + [None] * len(images))[: len(images)]
    return found


def _load_scanned(scanned: ScannedFile, info: object) -> _Loaded:
    """Build the frame record and decode its analysis preview.

    Inputs: scanned, the file from the folder scan; info, the capture-time record, or any other object.
    Output: a loaded frame with the image, or with an error string when the preview cannot be read.
    Uses: the preview workers.
    Limitations: the photograph is opened for reading only. The preview is an embedded JPEG or a paired JPEG, then resized to a 1024 px long edge. There is no demosaic.
    """

    frame = _frame_for(scanned, info)
    try:
        image = _load_analysis_image(scanned)
    except PreviewError as exc:
        log.warning("%s", exc)
        return _Loaded(scanned, frame, None, str(exc))
    except Exception as exc:
        log.warning("Could not read %s", scanned.path.name, exc_info=True)
        return _Loaded(scanned, frame, None, f"{exc.__class__.__name__}: {exc}")
    return _Loaded(scanned, frame, image, None)


def _score_loaded(
    scanned: ScannedFile,
    info: object,
    cache: Cache | None,
    weights: Weights,
    subject_key: str,
    detection: object,
    detection_failures: int,
    eye: object,
) -> tuple[FrameAnalysis, int]:
    """Decode one file and score it with a detection result that is already known.

    Inputs: scanned and info, the file and its capture clock; cache, weights, and subject_key, the measurement settings; detection, an optional subject box; detection_failures, 0 or 1; eye, an optional eye already found on the coordinating thread.
    Output: the analysis and the same detection-failure count.
    Uses: score_at when no model is running, passing an empty detection and no eye.
    Limitations: this does not write the SQLite cache. The preview is not demosaiced. An eye smaller than 16 px is stored without sharpness.
    """

    loaded = _load_scanned(scanned, info)
    return _score_image(
        loaded,
        loaded.image,
        detection,
        detection_failures,
        eye,
        cache,
        weights,
        subject_key,
    )


def _score_image(
    loaded: _Loaded,
    image: np.ndarray | None,
    detection: object,
    detection_failures: int,
    eye: object,
    cache: Cache | None,
    weights: Weights,
    subject_key: str,
) -> tuple[FrameAnalysis, int]:
    """Measure one decoded preview and write its cache JPEGs.

    Inputs: loaded, the frame and any decode error; image, the BGR preview or None; detection, an optional box with label and confidence; detection_failures, 0 or 1; eye, an optional eye from the pose model; cache, weights, and subject_key, the measurement settings.
    Output: a FrameAnalysis and the detection-failure count. A missing image becomes a failed analysis.
    Uses: the preview workers after a model batch, and the path with no model.
    Limitations: preview JPEGs go into the cache directory, not beside the photograph. This function does not search for an eye. Sharpness is None when the eye is smaller than 16 px. There is no demosaic. The SQLite cache row is written later on the coordinating thread.
    """

    if image is None:
        return _failed(loaded.frame, loaded.error or "No preview"), detection_failures
    try:
        box = getattr(detection, "box", None)
        metrics = score_bgr(
            image,
            subject_box=box,
            subject_label=getattr(detection, "label", None),
            subject_confidence=getattr(detection, "confidence", None),
        )
        if eye is not None:
            metrics = replace(
                metrics,
                eye_sharpness=getattr(eye, "sharpness", None),
                eye_center=(int(eye.x), int(eye.y)),
                eye_radius=int(eye.radius),
            )
        preview_path, thumb_path = _write_previews(cache, loaded.scanned, subject_key, image)
        return (
            FrameAnalysis(
                frame=loaded.frame,
                metrics=metrics,
                composite=composite_score(metrics, weights),
                preview_path=preview_path,
                thumb_path=thumb_path,
                error=None,
            ),
            detection_failures,
        )
    except PreviewError as exc:
        log.warning("%s", exc)
        return _failed(loaded.frame, str(exc)), detection_failures
    except Exception as exc:
        log.warning("Could not score %s", loaded.scanned.path.name, exc_info=True)
        return _failed(loaded.frame, f"{exc.__class__.__name__}: {exc}"), detection_failures


def _frame_for(scanned: ScannedFile, info: object) -> Frame:
    """Build a Frame from a scanned file and its capture-time record.

    Inputs: scanned, the path, kind, size, and mtime; info, an object with captured_at, source, and camera, or anything else.
    Output: a Frame. A missing or non-datetime clock stays None, and the timestamp source defaults to `missing`.
    Uses: preview loading before a frame is scored.
    Limitations: the file is not opened. Camera is kept only when it is a string.
    """

    captured_at = getattr(info, "captured_at", None)
    source = getattr(info, "source", None) or "missing"
    camera = getattr(info, "camera", None)
    from datetime import datetime

    moment = captured_at if isinstance(captured_at, datetime) else None
    return Frame(
        path=scanned.path,
        kind=scanned.kind,
        captured_at=moment,
        timestamp_source=str(source),
        camera=camera if isinstance(camera, str) else None,
        size=scanned.size,
        mtime_ns=scanned.mtime_ns,
        paired_jpeg=scanned.paired_jpeg,
    )


def _store_analysis(cache: Cache | None, analysis: FrameAnalysis, subject_key: str) -> None:
    """Write one scored frame into the SQLite cache.

    Inputs: cache, the cache or None; analysis, the scored frame; subject_key, the model and eye identity.
    Output: none. A complete analysis is stored. Incomplete rows are skipped.
    Uses: persist, on the coordinating thread.
    Limitations: nothing is written when the cache is missing or the analysis has no metrics or preview paths. This is not called from a worker thread. The original photograph is not written.
    """

    if cache is None or analysis.metrics is None or analysis.preview_path is None or analysis.thumb_path is None:
        return
    frame = analysis.frame
    cache.put(
        frame.path,
        frame.mtime_ns,
        frame.size,
        subject_key,
        captured_at=frame.captured_at,
        timestamp_source=frame.timestamp_source,
        camera=frame.camera,
        metrics=analysis.metrics,
        preview_path=analysis.preview_path,
        thumb_path=analysis.thumb_path,
    )


def _load_analysis_image(scanned: ScannedFile):
    """Decode a preview and shrink its long edge to 1024 px.

    Inputs: scanned, the photograph and its optional paired JPEG.
    Output: the BGR analysis image.
    Uses: _load_scanned.
    Limitations: there is no demosaic. A paired JPEG is tried first. If that decode fails, the embedded RAW preview is tried. The original is opened for reading only.
    """

    try:
        blob = extract_preview_bytes(scanned.path, scanned.paired_jpeg)
        image = decode_image(blob)
    except PreviewError:
        if scanned.paired_jpeg is None:
            raise
        log.info("Paired JPEG for %s could not be decoded; trying the RAW preview", scanned.path.name)
        blob = extract_preview_bytes(scanned.path, None)
        image = decode_image(blob)
    return resize_max_edge(image, ANALYSIS_EDGE)


def _write_previews(
    cache: Cache | None,
    scanned: ScannedFile,
    subject_key: str,
    image,
) -> tuple[Path | None, Path | None]:
    """Write the analysis JPEG and the 360 px thumbnail into the cache directory.

    Inputs: cache, the cache or None; scanned, the source identity; subject_key, part of the file digest; image, the analysis preview.
    Output: the preview path and thumbnail path, or None and None when there is no cache.
    Uses: _score_image after a frame is measured.
    Limitations: these are cache JPEGs, not the original photograph, which is never opened for writing. The SQLite cache row is a separate write on the coordinating thread.
    """

    if cache is None:
        return None, None
    digest = hashlib.sha1(
        f"{scanned.path}|{scanned.mtime_ns}|{scanned.size}|{subject_key}".encode("utf-8", "surrogateescape")
    ).hexdigest()[:24]
    preview_path = cache.directory / "previews" / f"{digest}.jpg"
    thumb_path = cache.directory / "thumbs" / f"{digest}.jpg"
    preview_path.write_bytes(encode_jpeg(image, quality=90))
    thumb_path.write_bytes(encode_jpeg(resize_max_edge(image, THUMB_EDGE), quality=80))
    return preview_path, thumb_path


def _from_cache(
    cache: Cache | None,
    scanned: ScannedFile,
    weights: Weights,
    subject_key: str,
) -> FrameAnalysis | None:
    """Rebuild an analysis from a cache row when the file identity still matches.

    Inputs: cache, the cache or None; scanned, the current path, size, and mtime; weights, the blend used to recompute the composite; subject_key, the model and eye identity.
    Output: a FrameAnalysis, or None when the cache is missing or has no matching row.
    Uses: _score_all before a file is measured again.
    Limitations: the photograph is not reread. A changed size, mtime, or subject key misses. The composite is recomputed from the stored metrics.
    """

    if cache is None:
        return None
    record = cache.get(scanned.path, scanned.mtime_ns, scanned.size, subject_key)
    if record is None:
        return None
    frame = Frame(
        path=scanned.path,
        kind=scanned.kind,
        captured_at=record.captured_at,
        timestamp_source=record.timestamp_source,
        camera=record.camera,
        size=scanned.size,
        mtime_ns=scanned.mtime_ns,
        paired_jpeg=scanned.paired_jpeg,
    )
    return FrameAnalysis(
        frame=frame,
        metrics=record.metrics,
        composite=composite_score(record.metrics, weights),
        preview_path=record.preview_path,
        thumb_path=record.thumb_path,
        error=None,
    )


def _failed(frame: Frame, message: str) -> FrameAnalysis:
    """Build an analysis that records why a preview could not be scored.

    Inputs: frame, the photograph; message, the error text.
    Output: a FrameAnalysis with no metrics, composite, or preview paths.
    Uses: the scoring path when decode or measurement fails.
    Limitations: the frame stays in the library so grouping can still place it. Nothing is written to the photograph.
    """

    return FrameAnalysis(
        frame=frame,
        metrics=None,
        composite=None,
        preview_path=None,
        thumb_path=None,
        error=message,
    )


def _retarget(item: FrameAnalysis, weights: Weights) -> FrameAnalysis:
    """Replace one frame's composite with the score for a new weight mix.

    Inputs: item, an existing analysis; weights, the weights to apply.
    Output: the same analysis with a new composite, or a None composite when metrics are missing.
    Uses: with_weights.
    Limitations: measurements are not recomputed and the photograph is not read.
    """

    if item.metrics is None:
        return replace(item, composite=None)
    return replace(item, composite=composite_score(item.metrics, weights))


def _check_cancel(should_cancel: Callable[[], bool] | None) -> None:
    """Raise when the photographer has cancelled the scan.

    Inputs: should_cancel, a callback, or None.
    Output: none. CancelledError is raised when the callback returns true.
    Uses: the cache walk and the measurement loops.
    Limitations: a missing callback does nothing. Work already finished is not undone.
    """

    if should_cancel is not None and should_cancel():
        raise CancelledError("Scan cancelled")


def _report(progress: ProgressCallback | None, done: int, total: int, message: str) -> None:
    """Forward a progress line when a callback was provided.

    Inputs: progress, the callback or None; done and total, the counts; message, the status text.
    Output: none. The callback is called only when it is set.
    Uses: analyze_folder and the scoring loops.
    Limitations: this does not write a file and does not decide whether the scan is cancelled.
    """

    if progress is not None:
        progress(done, total, message)
