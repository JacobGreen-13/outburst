"""Timestamp reading, including the built-in EXIF parser."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from outburst.metadata import (
    SOURCE_EXIF,
    SOURCE_EXIFREAD,
    SOURCE_EXIFTOOL,
    SOURCE_HEADER,
    _pack_tiff,
    build_exif_app1,
    inject_exif,
    read_capture_info,
    timestamp_reader_warning,
)
from tests.helpers import sharp_frame, write_jpeg


def test_jpeg_exif_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "bird.jpg"
    when = datetime(2024, 5, 6, 7, 8, 9, 200000)
    write_jpeg(path, sharp_frame(), when, model="BirdCam")
    info = read_capture_info([path])[path]
    assert info.captured_at == when
    assert info.source == SOURCE_EXIFREAD
    assert info.camera == "BirdCam"
    decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None


def test_inline_and_offset_subseconds(tmp_path: Path) -> None:
    short = tmp_path / "short.jpg"
    long = tmp_path / "long.jpg"
    image = sharp_frame()
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    base = encoded.tobytes()
    short.write_bytes(inject_exif(base, datetime(2024, 1, 2, 3, 4, 5), "25", "Cam"))
    long.write_bytes(inject_exif(base, datetime(2024, 1, 2, 3, 4, 6), "4256", "Cam"))
    short_info = read_capture_info([short])[short]
    long_info = read_capture_info([long])[long]
    assert short_info.captured_at == datetime(2024, 1, 2, 3, 4, 5, 250000)
    assert long_info.captured_at == datetime(2024, 1, 2, 3, 4, 6, 425600)


def test_big_endian_and_raw_magic_numbers() -> None:
    when = datetime(2020, 2, 3, 4, 5, 6)
    for magic, endian in ((42, ">"), (0x4F52, "<"), (0x0055, "<")):
        segment = build_exif_app1(when, "7", "ORFCam", endian=endian, magic=magic)
        payload = segment[4 + 6 :]  # skip marker, length, and Exif\x00\x00
        # build_exif_app1 returns the APP1 marker. The TIFF header follows Exif\x00\x00.
        assert b"Exif\x00\x00" in segment
        tiff = segment.split(b"Exif\x00\x00", 1)[1]
        from outburst.metadata import _from_tiff_bytes

        info = _from_tiff_bytes(tiff)
        assert info is not None
        assert info.captured_at == datetime(2020, 2, 3, 4, 5, 6, 700000)
        assert info.camera == "ORFCam"
        del payload


def test_header_scan_reads_a_cr3_like_file(tmp_path: Path) -> None:
    path = tmp_path / "bird.CR3"
    path.write_bytes(b"\x00\x00\x00\x18ftypcrx " + b" pad " + b"2024:07:08 09:10:11" + b" tail")
    info = read_capture_info([path])[path]
    assert info.captured_at == datetime(2024, 7, 8, 9, 10, 11)
    assert info.source == SOURCE_HEADER


def test_missing_clock_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "empty.jpg"
    ok, encoded = cv2.imencode(".jpg", np.full((16, 16, 3), 80, dtype=np.uint8))
    assert ok
    path.write_bytes(encoded.tobytes())
    info = read_capture_info([path])[path]
    assert info.captured_at is None
    assert info.source == "missing"


def test_exiftool_process_is_hidden_on_windows(monkeypatch) -> None:
    from outburst import metadata

    assert metadata._hidden_windows_kwargs() == {}

    class StartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0
            self.wShowWindow = None

    monkeypatch.setattr(metadata.sys, "platform", "win32")
    monkeypatch.setattr(metadata.subprocess, "STARTUPINFO", StartupInfo, raising=False)
    monkeypatch.setattr(metadata.subprocess, "STARTF_USESHOWWINDOW", 0x1, raising=False)
    monkeypatch.setattr(metadata.subprocess, "SW_HIDE", 0, raising=False)
    monkeypatch.setattr(metadata.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    flags = metadata._hidden_windows_kwargs()
    info = flags["startupinfo"]
    assert isinstance(info, StartupInfo)
    assert info.dwFlags & 0x1
    assert info.wShowWindow == 0
    assert flags["creationflags"] == 0x08000000


def test_exiftool_json_is_preferred(monkeypatch, tmp_path: Path) -> None:
    path = (tmp_path / "a.jpg").resolve()
    path.write_bytes(b"not a photo")
    payload = json.dumps(
        [
            {
                "SourceFile": str(path),
                "DateTimeOriginal": "2024:01:02 03:04:05",
                "SubSecTimeOriginal": "25",
                "Model": "BirdCam",
            }
        ]
    )

    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/exiftool" if name == "exiftool" else None)

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        del command, kwargs
        return SimpleNamespace(returncode=0, stdout=payload.encode(), stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def exifread_must_not_run(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise AssertionError("exiftool already supplied the timestamp")

    import exifread

    monkeypatch.setattr(exifread, "process_file", exifread_must_not_run)
    info = read_capture_info([path])[path]
    assert info.source == SOURCE_EXIFTOOL
    assert info.captured_at == datetime(2024, 1, 2, 3, 4, 5, 250000)
    assert info.camera == "BirdCam"


def test_exifread_reads_a_tiff_based_raw(tmp_path: Path) -> None:
    path = tmp_path / "bird.CR2"
    when = datetime(2020, 2, 3, 4, 5, 6, 700000)
    path.write_bytes(_pack_tiff(when, "7", "Cam", endian="<", magic=42))
    info = read_capture_info([path])[path]
    assert info.source == SOURCE_EXIFREAD
    assert info.captured_at == when
    assert info.camera == "Cam"


def test_exifread_error_falls_back_to_the_builtin_reader(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "bird.jpg"
    when = datetime(2024, 5, 6, 7, 8, 9, 200000)
    write_jpeg(path, sharp_frame(), when, model="BirdCam")

    import exifread

    def boom(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("exifread failed")

    monkeypatch.setattr(exifread, "process_file", boom)
    info = read_capture_info([path])[path]
    assert info.captured_at == when
    assert info.source == SOURCE_EXIF
    assert info.camera == "BirdCam"


def test_reader_warning_names_unrecognized_raw_formats(monkeypatch) -> None:
    monkeypatch.setattr("outburst.metadata.exiftool_available", lambda: False)
    monkeypatch.setattr("outburst.metadata.exifread_available", lambda: True)
    text = timestamp_reader_warning([".CR3", "raf", ".cr2"])
    assert text is not None
    assert "exifread" in text
    assert "CR2" in text
    assert "This folder includes CR3, RAF." in text
    assert timestamp_reader_warning([]) is None


def test_reader_warning_is_quiet_when_exiftool_is_installed(monkeypatch) -> None:
    monkeypatch.setattr("outburst.metadata.exiftool_available", lambda: True)
    assert timestamp_reader_warning([".cr3"]) is None


def test_reader_warning_without_exifread(monkeypatch) -> None:
    monkeypatch.setattr("outburst.metadata.exiftool_available", lambda: False)
    monkeypatch.setattr("outburst.metadata.exifread_available", lambda: False)
    text = timestamp_reader_warning([".nef"])
    assert text is not None
    assert "built-in" in text
    assert "CR3" in text
