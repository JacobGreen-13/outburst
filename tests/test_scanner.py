"""Folder scanning and RAW/JPEG pairing."""

from __future__ import annotations

from pathlib import Path

import pytest

from outburst.scanner import scan_folder


def test_pairs_a_jpeg_with_the_matching_raw_and_skips_hidden_files(tmp_path: Path) -> None:
    (tmp_path / "IMG_0001.CR3").write_bytes(b"raw")
    (tmp_path / "IMG_0001.JPG").write_bytes(b"jpeg")
    (tmp_path / "solo.jpeg").write_bytes(b"jpeg")
    (tmp_path / "notes.txt").write_bytes(b"nope")
    (tmp_path / "IMG_0001.xmp").write_bytes(b"<xmp>")
    (tmp_path / ".DS_Store").write_bytes(b"hidden")
    hidden = tmp_path / ".secret"
    hidden.mkdir()
    (hidden / "hidden.RAF").write_bytes(b"raw")

    found = scan_folder(tmp_path, recursive=False)
    by_name = {item.path.name: item for item in found}
    assert set(by_name) == {"IMG_0001.CR3", "solo.jpeg"}
    assert by_name["IMG_0001.CR3"].kind == "raw"
    assert by_name["IMG_0001.CR3"].paired_jpeg is not None
    assert by_name["IMG_0001.CR3"].paired_jpeg.name == "IMG_0001.JPG"
    assert by_name["solo.jpeg"].kind == "jpeg"
    assert by_name["solo.jpeg"].paired_jpeg is None


def test_recursive_scan_includes_nested_raw_files(tmp_path: Path) -> None:
    nested = tmp_path / "day1"
    nested.mkdir()
    (nested / "bird.NEF").write_bytes(b"raw")
    (tmp_path / "top.ARW").write_bytes(b"raw")
    names = {item.path.name for item in scan_folder(tmp_path, recursive=True)}
    assert names == {"bird.NEF", "top.ARW"}
    assert scan_folder(tmp_path, recursive=False)[0].path.name == "top.ARW"


def test_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        scan_folder(tmp_path / "missing")
