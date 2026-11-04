"""Embedded preview extraction."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from outburst.preview import PreviewError, decode_image, extract_preview_bytes
import outburst.preview as preview_module
from tests.helpers import sharp_frame


def _jpeg_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", sharp_frame())
    assert ok
    return encoded.tobytes()


def test_scans_an_embedded_jpeg_when_rawpy_cannot(tmp_path: Path, monkeypatch) -> None:
    blob = _jpeg_bytes()
    raw = tmp_path / "bird.CR2"
    raw.write_bytes(b"not-a-real-raw" + blob + b"trailer")
    monkeypatch.setattr(preview_module, "rawpy", None)
    extracted = extract_preview_bytes(raw)
    decoded = decode_image(extracted)
    assert decoded.shape[0] > 10
    assert decoded.shape[2] == 3


def test_rawpy_jpeg_thumbnail_is_preferred(tmp_path: Path, monkeypatch) -> None:
    blob = _jpeg_bytes()
    raw = tmp_path / "bird.NEF"
    raw.write_bytes(b"raw-body")

    class ThumbFormat:
        JPEG = 1
        BITMAP = 2

    class Thumb:
        format = ThumbFormat.JPEG
        data = blob

    class Raw:
        def extract_thumb(self) -> Thumb:
            return Thumb()

        def __enter__(self) -> Raw:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    fake = type("rawpy", (), {"imread": lambda _path: Raw(), "ThumbFormat": ThumbFormat})
    monkeypatch.setattr(preview_module, "rawpy", fake)
    assert extract_preview_bytes(raw) == blob


def test_rawpy_bitmap_thumbnail_is_encoded(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "bird.ARW"
    raw.write_bytes(b"raw-body")
    rgb = np.zeros((24, 32, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255

    class ThumbFormat:
        JPEG = 1
        BITMAP = 2

    class Thumb:
        format = ThumbFormat.BITMAP
        data = rgb

    class Raw:
        def extract_thumb(self) -> Thumb:
            return Thumb()

        def __enter__(self) -> Raw:
            return self

        def __exit__(self, *args: object) -> bool:
            return False

    fake = type("rawpy", (), {"imread": lambda _path: Raw(), "ThumbFormat": ThumbFormat})
    monkeypatch.setattr(preview_module, "rawpy", fake)
    decoded = decode_image(extract_preview_bytes(raw))
    assert decoded.shape[:2] == (24, 32)


def test_paired_jpeg_is_used_before_the_raw(tmp_path: Path) -> None:
    blob = _jpeg_bytes()
    raw = tmp_path / "bird.DNG"
    raw.write_bytes(b"raw")
    paired = tmp_path / "bird.JPG"
    paired.write_bytes(blob)
    assert extract_preview_bytes(raw, paired) == blob


def test_missing_preview_raises(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "bird.RAF"
    raw.write_bytes(b"no jpeg here")
    monkeypatch.setattr(preview_module, "rawpy", None)
    try:
        extract_preview_bytes(raw)
    except PreviewError as exc:
        assert "bird.RAF" in str(exc)
    else:
        raise AssertionError("expected PreviewError")
