"""Load a small preview for scoring and display.

RAW files are never demosaiced. Outburst asks rawpy for the embedded JPEG
and, if that is missing, scans the file header for an embedded JPEG. A JPEG
that shares the RAW file's name is preferred when the camera wrote one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

try:
    import rawpy
except ImportError:  # pragma: no cover - rawpy is a required dependency
    rawpy = None  # type: ignore[assignment]

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_READ_LIMIT = 48 * 1024 * 1024


class PreviewError(RuntimeError):
    """The preview could not be read. The original file was not modified."""


def extract_preview_bytes(path: Path, paired_jpeg: Path | None = None) -> bytes:
    """Return JPEG (or JPEG-encoded bitmap) bytes for a photograph."""

    if paired_jpeg is not None:
        try:
            data = paired_jpeg.read_bytes()
        except OSError as exc:
            log.info("Paired JPEG %s could not be read: %s", paired_jpeg, exc)
        else:
            if data:
                return data

    if path.suffix.lower() in {".jpg", ".jpeg"}:
        try:
            return path.read_bytes()
        except OSError as exc:
            raise PreviewError(f"Could not read {path.name}: {exc}") from exc

    rawpy_error: str | None = None
    if rawpy is not None:
        try:
            encoded = _extract_with_rawpy(path)
        except Exception as exc:
            rawpy_error = f"{exc.__class__.__name__}: {exc}"
            log.info("rawpy preview failed for %s: %s", path.name, exc)
        else:
            if encoded:
                return encoded

    scanned = _scan_embedded_jpeg(path)
    if scanned is not None:
        return scanned

    detail = f" ({rawpy_error})" if rawpy_error else ""
    raise PreviewError(f"No embedded preview in {path.name}{detail}")


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG bytes into a BGR array."""

    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise PreviewError("Preview could not be decoded")
    return image


def resize_max_edge(image: np.ndarray, max_edge: int) -> np.ndarray:
    """Shrink so the long edge is at most `max_edge`. Smaller images stay as they are."""

    height, width = image.shape[:2]
    long_edge = max(height, width)
    if long_edge <= max_edge or long_edge == 0:
        return image
    scale = max_edge / long_edge
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality: int = 90) -> bytes:
    ok, buffer = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise PreviewError("Could not encode a preview")
    return buffer.tobytes()


def _extract_with_rawpy(path: Path) -> bytes | None:
    assert rawpy is not None
    with rawpy.imread(str(path)) as raw:
        thumb = raw.extract_thumb()
        if thumb.format == rawpy.ThumbFormat.JPEG:
            data = bytes(thumb.data) if thumb.data is not None else b""
            return data or None
        if thumb.format == rawpy.ThumbFormat.BITMAP:
            rgb = np.asarray(thumb.data)
            if rgb.size == 0:
                return None
            if rgb.ndim == 2:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            else:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return encode_jpeg(bgr)
    return None


def _scan_embedded_jpeg(path: Path) -> bytes | None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PreviewError(f"Could not read {path.name}: {exc}") from exc
    try:
        with path.open("rb") as handle:
            data = handle.read(min(size, _READ_LIMIT))
    except OSError as exc:
        raise PreviewError(f"Could not read {path.name}: {exc}") from exc

    candidates: list[bytes] = []
    start = 0
    while len(candidates) < 12:
        soi = data.find(_JPEG_SOI, start)
        if soi < 0:
            break
        eoi = data.find(_JPEG_EOI, soi + 2)
        if eoi < 0:
            break
        blob = data[soi : eoi + 2]
        start = soi + 2
        if len(blob) >= 4000:
            candidates.append(blob)
    candidates.sort(key=len, reverse=True)
    for blob in candidates[:6]:
        try:
            decode_image(blob)
        except PreviewError:
            continue
        return blob
    return None
