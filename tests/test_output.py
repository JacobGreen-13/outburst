"""Sidecars, keeper export, and reports leave the originals alone."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pytest

from outburst.models import Burst, Frame, FrameAnalysis, Library, Metrics, Weights
from outburst.output import (
    SidecarError,
    choose_keepers,
    export_keepers,
    upsert_xmp,
    write_library_sidecars,
    write_reports,
    write_xmp_sidecar,
)


def _frame(path: Path, composite: float) -> tuple[Frame, FrameAnalysis]:
    frame = Frame(path, "jpeg", datetime(2024, 6, 1, 12, 0, 0), "exif", "TestCam", path.stat().st_size, path.stat().st_mtime_ns)
    metrics = Metrics(composite * 100, composite * 80, 5, 0.9, 0.0, 0.0)
    analysis = FrameAnalysis(frame, metrics, composite, None, None, None)
    return frame, analysis


def _library(folder: Path, frames: list[tuple[Frame, FrameAnalysis]]) -> Library:
    burst = Burst("burst-001", tuple(frame for frame, _analysis in frames))
    return Library(
        folder=folder,
        analyses=tuple(analysis for _frame, analysis in frames),
        bursts=(burst,),
        warnings=(),
        threshold=1.0,
        weights=Weights(),
    )


def _digest(path: Path) -> bytes:
    return hashlib.sha256(path.read_bytes()).digest()


def test_xmp_does_not_touch_the_photograph(tmp_path: Path) -> None:
    photo = tmp_path / "IMG_0001.CR3"
    photo.write_bytes(b"original-raw-bytes")
    before = (_digest(photo), photo.stat().st_mtime_ns)
    frame, analysis = _frame(photo, 0.8)
    library = _library(tmp_path, [(frame, analysis)])
    written, warnings = write_library_sidecars(library, {})
    assert warnings == []
    assert written == [photo.with_suffix(".xmp")]
    text = written[0].read_text(encoding="utf-8")
    assert 'xmp:Rating="5"' in text
    assert "outburst:Score" in text
    assert "Outburst" in text
    assert (_digest(photo), photo.stat().st_mtime_ns) == before


def test_existing_xmp_keeps_unrelated_fields(tmp_path: Path) -> None:
    photo = tmp_path / "IMG_0002.NEF"
    photo.write_bytes(b"raw")
    sidecar = photo.with_suffix(".xmp")
    sidecar.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about="" xmlns:xmp="http://ns.adobe.com/xap/1.0/" xmp:Rating="3" xmp:Label="Blue">
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>
""",
        encoding="utf-8",
    )
    write_xmp_sidecar(photo, {"xmp:Rating": "5", "outburst:Score": "0.5000"})
    text = sidecar.read_text(encoding="utf-8")
    assert 'xmp:Label="Blue"' in text
    assert 'xmp:Rating="5"' in text
    assert 'outburst:Score="0.5000"' in text
    assert _digest(photo) == hashlib.sha256(b"raw").digest()


def test_refuses_to_overwrite_a_non_xmp_sidecar(tmp_path: Path) -> None:
    photo = tmp_path / "IMG.JPG"
    photo.write_bytes(b"jpeg")
    sidecar = photo.with_suffix(".xmp")
    sidecar.write_text("this is my notes file", encoding="utf-8")
    with pytest.raises(SidecarError):
        upsert_xmp(sidecar.read_text(encoding="utf-8"), {"xmp:Rating": "5"})
    assert sidecar.read_text(encoding="utf-8") == "this is my notes file"


def test_choose_keepers_and_export_modes(tmp_path: Path) -> None:
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"aaa")
    second.write_bytes(b"bbb")
    frame_a, analysis_a = _frame(first, 0.4)
    frame_b, analysis_b = _frame(second, 0.9)
    burst = Burst("burst-001", (frame_a, frame_b))
    by_path = {first: analysis_a, second: analysis_b}
    assert [choice.path for choice in choose_keepers([burst], by_path, {})] == [second]
    assert [choice.path for choice in choose_keepers([burst], by_path, {second: "reject"})] == [first]
    assert choose_keepers([burst], by_path, {first: "reject", second: "reject"}) == []
    marked = choose_keepers([burst], by_path, {first: "keep"})
    assert [choice.path for choice in marked] == [first]

    copied = export_keepers([second], tmp_path / "keepers", "copy")
    assert copied[0].method == "copy"
    assert copied[0].destination.read_bytes() == b"bbb"
    assert second.read_bytes() == b"bbb"

    linked = export_keepers([first], tmp_path / "links", "hardlink")
    assert linked[0].method == "hardlink"
    assert linked[0].destination.samefile(first)
    assert first.read_bytes() == b"aaa"


def test_report_lists_the_recommended_frame(tmp_path: Path) -> None:
    photo = tmp_path / "IMG_0009.jpg"
    photo.write_bytes(b"photo")
    frame, analysis = _frame(photo, 0.7)
    library = _library(tmp_path, [(frame, analysis)])
    json_path, csv_path = write_reports(library, {}, tmp_path / "reports")
    text = json_path.read_text(encoding="utf-8")
    assert "Outburst" in text
    assert "IMG_0009.jpg" in text
    assert "recommended" in csv_path.read_text(encoding="utf-8")
