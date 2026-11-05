"""Scan a folder, score previews, and group bursts.

This is the part the window calls. It reads photographs and writes only inside
the cache directory. Original files are opened for reading.
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
    progress: ProgressCallback | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Library:
    """Score every supported photograph in `folder` and group them into bursts."""

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
    if eye_scoring:
        subject_key = f"{subject_key}|eye" if subject_key else "eye"

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
            eye_scoring,
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
            eye_scoring=eye_scoring,
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
        eye_active=eye_scoring,
    )


def with_weights(library: Library, weights: Weights) -> Library:
    """Recompute composite scores from cached measurements."""

    weights = weights.clamped()
    analyses = tuple(_retarget(item, weights) for item in library.analyses)
    return replace(library, analyses=analyses, weights=weights)


def with_threshold(library: Library, threshold: float) -> Library:
    """Regroup the same frames under a new gap threshold."""

    bursts = tuple(group_frames([item.frame for item in library.analyses], threshold))
    return replace(library, bursts=bursts, threshold=float(threshold))


def _score_all(
    files: list[ScannedFile],
    *,
    cache: Cache | None,
    weights: Weights,
    detector: SubjectDetector | None,
    subject_key: str,
    eye_scoring: bool,
    progress: ProgressCallback | None,
    should_cancel: Callable[[], bool] | None,
) -> tuple[list[FrameAnalysis], int]:
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
            eye_scoring=eye_scoring,
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
    eye_scoring: bool,
    progress: ProgressCallback | None,
    done_offset: int,
    total: int,
    should_cancel: Callable[[], bool] | None,
) -> dict[Path, tuple[FrameAnalysis, int]]:
    """Score cache misses. Preview work uses a small pool. The cache is written here."""

    found: dict[Path, tuple[FrameAnalysis, int]] = {}

    def persist(index: int, pair: tuple[FrameAnalysis, int]) -> None:
        analysis, _failed_count = pair
        _store_analysis(cache, analysis, subject_key)
        found[analysis.frame.path] = pair
        _report(progress, done_offset + index + 1, total, f"Scored {pending[index].path.name}")

    def cancel() -> None:
        _check_cancel(should_cancel)

    if detector is None:

        def score_at(index: int) -> tuple[FrameAnalysis, int]:
            scanned = pending[index]
            return _score_loaded(
                scanned,
                capture_info.get(scanned.path),
                cache,
                weights,
                subject_key,
                eye_scoring,
                None,
                0,
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
        eye_scoring=eye_scoring,
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
    detector: SubjectDetector,
    subject_key: str,
    eye_scoring: bool,
    persist: Callable[[int, tuple[FrameAnalysis, int]], None],
    cancel: Callable[[], None],
) -> None:
    """Decode ahead on a pool, run the subject model on this thread, then score.

    The model is not shared across threads. A batch is a few frames. Decode of
    the next frames keeps running while that batch is predicted.
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
                boxes, failures = _detect_batch(detector, images)
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
                        cache,
                        weights,
                        subject_key,
                        eye_scoring,
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


def _load_scanned(scanned: ScannedFile, info: object) -> _Loaded:
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
    eye_scoring: bool,
    detection: object,
    detection_failures: int,
) -> tuple[FrameAnalysis, int]:
    loaded = _load_scanned(scanned, info)
    return _score_image(
        loaded,
        loaded.image,
        detection,
        detection_failures,
        cache,
        weights,
        subject_key,
        eye_scoring,
    )


def _score_image(
    loaded: _Loaded,
    image: np.ndarray | None,
    detection: object,
    detection_failures: int,
    cache: Cache | None,
    weights: Weights,
    subject_key: str,
    eye_scoring: bool,
) -> tuple[FrameAnalysis, int]:
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
        if eye_scoring:
            from outburst.eyes import locate_eye

            eye = locate_eye(image, box)
            if eye is not None:
                metrics = replace(
                    metrics,
                    eye_sharpness=eye.sharpness,
                    eye_center=(eye.x, eye.y),
                    eye_radius=eye.radius,
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
    return FrameAnalysis(
        frame=frame,
        metrics=None,
        composite=None,
        preview_path=None,
        thumb_path=None,
        error=message,
    )


def _retarget(item: FrameAnalysis, weights: Weights) -> FrameAnalysis:
    if item.metrics is None:
        return replace(item, composite=None)
    return replace(item, composite=composite_score(item.metrics, weights))


def _check_cancel(should_cancel: Callable[[], bool] | None) -> None:
    if should_cancel is not None and should_cancel():
        raise CancelledError("Scan cancelled")


def _report(progress: ProgressCallback | None, done: int, total: int, message: str) -> None:
    if progress is not None:
        progress(done, total, message)
