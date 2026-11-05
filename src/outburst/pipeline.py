"""Scan a folder, score previews, and group bursts.

This is the part the window calls. It reads photographs and writes only inside
the cache directory. Original files are opened for reading.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from outburst.cache import Cache, default_cache_dir
from outburst.grouping import group_frames
from outburst.metadata import read_capture_info, timestamp_reader_warning
from outburst.models import Frame, FrameAnalysis, Library, Weights
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
    for scanned in files:
        _check_cancel(should_cancel)
        cached = _from_cache(cache, scanned, weights, subject_key)
        if cached is not None:
            cached_rows[scanned.path] = cached
        else:
            pending.append(scanned)

    if pending:
        _report(progress, 0, total, "Reading capture times")
        capture_info = read_capture_info([item.path for item in pending])
    else:
        capture_info = {}

    detection_failures = 0
    analyses: list[FrameAnalysis] = []
    for index, scanned in enumerate(files, start=1):
        _check_cancel(should_cancel)
        cached = cached_rows.get(scanned.path)
        if cached is not None:
            analyses.append(cached)
            _report(progress, index, total, f"Cached {scanned.path.name}")
            continue
        _report(progress, index - 1, total, f"Scoring {scanned.path.name}")
        info = capture_info.get(scanned.path)
        analysis, failed = _score_file(
            scanned,
            captured_at=None if info is None else info.captured_at,
            timestamp_source="missing" if info is None else info.source,
            camera=None if info is None else info.camera,
            cache=cache,
            weights=weights,
            detector=detector,
            subject_key=subject_key,
            eye_scoring=eye_scoring,
        )
        detection_failures += failed
        analyses.append(analysis)
        _report(progress, index, total, f"Scored {scanned.path.name}")
    return analyses, detection_failures


def _score_file(
    scanned: ScannedFile,
    *,
    captured_at: object,
    timestamp_source: str,
    camera: str | None,
    cache: Cache | None,
    weights: Weights,
    detector: SubjectDetector | None,
    subject_key: str,
    eye_scoring: bool,
) -> tuple[FrameAnalysis, int]:
    from datetime import datetime

    moment = captured_at if isinstance(captured_at, datetime) else None
    frame = Frame(
        path=scanned.path,
        kind=scanned.kind,
        captured_at=moment,
        timestamp_source=timestamp_source,
        camera=camera,
        size=scanned.size,
        mtime_ns=scanned.mtime_ns,
        paired_jpeg=scanned.paired_jpeg,
    )
    detection_failures = 0
    try:
        image = _load_analysis_image(scanned)
        detection = None
        if detector is not None:
            try:
                detection = detector.detect(image)
            except Exception:
                log.warning("Subject detection failed for %s", scanned.path.name, exc_info=True)
                detection_failures = 1
        metrics = score_bgr(
            image,
            subject_box=None if detection is None else detection.box,
            subject_label=None if detection is None else detection.label,
            subject_confidence=None if detection is None else detection.confidence,
        )
        if eye_scoring:
            from outburst.eyes import locate_eye

            eye = locate_eye(image, None if detection is None else detection.box)
            if eye is not None:
                metrics = replace(
                    metrics,
                    eye_sharpness=eye.sharpness,
                    eye_center=(eye.x, eye.y),
                    eye_radius=eye.radius,
                )
        preview_path, thumb_path = _write_previews(cache, scanned, subject_key, image)
        if cache is not None and preview_path is not None and thumb_path is not None:
            cache.put(
                scanned.path,
                scanned.mtime_ns,
                scanned.size,
                subject_key,
                captured_at=moment,
                timestamp_source=timestamp_source,
                camera=camera,
                metrics=metrics,
                preview_path=preview_path,
                thumb_path=thumb_path,
            )
        return (
            FrameAnalysis(
                frame=frame,
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
        return _failed(frame, str(exc)), detection_failures
    except Exception as exc:
        log.warning("Could not score %s", scanned.path, exc_info=True)
        return _failed(frame, f"{exc.__class__.__name__}: {exc}"), detection_failures


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
