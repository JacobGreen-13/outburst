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
    review_xmp_fields,
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
    partner = tmp_path / "IMG_0002.CR3"
    partner.write_bytes(b"partner-raw-bytes")
    before = (_digest(photo), photo.stat().st_mtime_ns)
    frame, analysis = _frame(photo, 0.8)
    partner_frame, partner_analysis = _frame(partner, 0.2)
    library = _library(tmp_path, [(frame, analysis), (partner_frame, partner_analysis)])
    written, warnings = write_library_sidecars(library, {photo: "keep"})
    assert warnings == []
    sidecar = photo.with_suffix(".xmp")
    assert sidecar in written
    text = sidecar.read_text(encoding="utf-8")
    assert 'xmp:Rating="5"' in text
    assert "outburst:Score" in text
    assert "Outburst" in text
    assert (_digest(photo), photo.stat().st_mtime_ns) == before
    assert not partner.with_suffix(".xmp").exists()


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
    assert choose_keepers([burst], by_path, {}) == []
    assert choose_keepers([burst], by_path, {second: "reject"}) == []
    assert choose_keepers([burst], by_path, {first: "reject", second: "reject"}) == []
    marked = choose_keepers([burst], by_path, {first: "keep"})
    assert [choice.path for choice in marked] == [first]
    assert marked[0].reason == "marked"
    both = choose_keepers([burst], by_path, {first: "keep", second: "keep"})
    assert [choice.path for choice in both] == [second, first]
    third = tmp_path / "c.jpg"
    third.write_bytes(b"ccc")
    frame_c, analysis_c = _frame(third, 0.5)
    by_path[third] = analysis_c
    single = Burst("burst-002", (frame_c,))
    assert choose_keepers([single], by_path, {third: "keep"}) == []
    shown = choose_keepers([single], by_path, {third: "keep"}, minimum_frames=1)
    assert [choice.path for choice in shown] == [third]
    mixed = choose_keepers([burst, single], by_path, {first: "keep", third: "keep"})
    assert [choice.path for choice in mixed] == [first]
    assert choose_keepers([burst], by_path, {first: "keep"}, minimum_frames=3) == []

    copied = export_keepers([second], tmp_path / "keepers", "copy")
    assert copied[0].method == "copy"
    assert copied[0].destination.read_bytes() == b"bbb"
    assert second.read_bytes() == b"bbb"

    linked = export_keepers([first], tmp_path / "links", "hardlink")
    assert linked[0].method == "hardlink"
    assert linked[0].destination.samefile(first)
    assert first.read_bytes() == b"aaa"

    names = []
    for index in range(4):
        path = tmp_path / f"keep_{index}.jpg"
        path.write_bytes(f"body-{index}".encode())
        names.append(path)
    copied_many = export_keepers(names, tmp_path / "batch", "copy")
    assert [item.source for item in copied_many] == names
    assert [item.destination.name for item in copied_many] == [path.name for path in names]
    assert copied_many[3].destination.read_bytes() == b"body-3"


def test_sidecar_writes_stars_and_pick_flags(tmp_path: Path) -> None:
    photo = tmp_path / "IMG_0004.CR2"
    photo.write_bytes(b"raw")
    frame, analysis = _frame(photo, 0.6)
    partner = tmp_path / "IMG_0005.CR2"
    partner.write_bytes(b"raw-2")
    partner_frame, partner_analysis = _frame(partner, 0.3)
    library = _library(tmp_path, [(frame, analysis), (partner_frame, partner_analysis)])
    written, warnings = write_library_sidecars(
        library,
        {photo: "keep", partner: "reject"},
        stars={photo: 3, partner: 4},
    )
    assert warnings == []
    kept = next(path for path in written if path.name.startswith("IMG_0004")).read_text(encoding="utf-8")
    rejected = next(path for path in written if path.name.startswith("IMG_0005")).read_text(encoding="utf-8")
    assert 'xmp:Rating="3"' in kept
    assert 'MicrosoftPhoto:Rating="50"' in kept
    assert 'xmpDM:pick="1"' in kept
    assert 'xmpDM:good="true"' in kept
    assert 'digiKam:PickLabel="3"' in kept
    assert "http://ns.adobe.com/xmp/1.0/DynamicMedia/" in kept
    assert 'xmp:Rating="4"' in rejected
    assert 'xmpDM:pick="-1"' in rejected
    assert 'xmpDM:good="false"' in rejected
    assert 'digiKam:PickLabel="1"' in rejected
    assert _digest(photo) == hashlib.sha256(b"raw").digest()

    bare = review_xmp_fields("reject", None)
    assert bare["xmp:Rating"] == "-1"
    assert bare["xmpDM:pick"] == "-1"
    cleared = review_xmp_fields(None, None)
    assert cleared["xmp:Rating"] == "0"
    assert cleared["xmpDM:pick"] == "0"
    assert cleared["xmpDM:good"] is None

    write_xmp_sidecar(photo, review_xmp_fields(None, 2))
    again = photo.with_suffix(".xmp").read_text(encoding="utf-8")
    assert 'xmp:Rating="2"' in again
    assert 'xmpDM:pick="0"' in again
    assert "xmpDM:good" not in again
    assert again.count("<rdf:Description") == 1
    assert "/>" in again


def test_unmarked_frame_is_not_rated_as_a_keeper(tmp_path: Path) -> None:
    photo = tmp_path / "IMG_0003.JPG"
    photo.write_bytes(b"jpeg")
    frame, analysis = _frame(photo, 0.8)
    library = _library(tmp_path, [(frame, analysis)])
    written, warnings = write_library_sidecars(library, {})
    assert warnings == []
    assert written == []
    assert not photo.with_suffix(".xmp").exists()
    assert _digest(photo) == hashlib.sha256(b"jpeg").digest()


def test_sidecars_cover_keepers_and_star_ratings_only(tmp_path: Path) -> None:
    keeper = tmp_path / "keep.CR3"
    starred = tmp_path / "stars.CR3"
    rejected = tmp_path / "reject.CR3"
    plain = tmp_path / "plain.CR3"
    for path, body in ((keeper, b"k"), (starred, b"s"), (rejected, b"r"), (plain, b"p")):
        path.write_bytes(body)
    library = _library(
        tmp_path,
        [_frame(keeper, 0.9), _frame(starred, 0.5), _frame(rejected, 0.4), _frame(plain, 0.2)],
    )
    plain_xmp = plain.with_suffix(".xmp")
    plain_xmp.write_text("camera notes", encoding="utf-8")
    written, warnings = write_library_sidecars(
        library,
        {keeper: "keep", rejected: "reject"},
        stars={starred: 2},
    )
    assert warnings == []
    assert {path.name for path in written} == {"keep.xmp", "stars.xmp"}
    assert not rejected.with_suffix(".xmp").exists()
    assert plain_xmp.read_text(encoding="utf-8") == "camera notes"
    starred_text = starred.with_suffix(".xmp").read_text(encoding="utf-8")
    assert 'xmp:Rating="2"' in starred_text
    assert 'xmpDM:pick="0"' in starred_text
    assert "xmpDM:good" not in starred_text
    assert _digest(keeper) == hashlib.sha256(b"k").digest()


def test_short_burst_stars_are_written_without_a_pick(tmp_path: Path) -> None:
    photo = tmp_path / "solo.CR3"
    photo.write_bytes(b"raw")
    library = _library(tmp_path, [_frame(photo, 0.8)])
    written, warnings = write_library_sidecars(library, {photo: "keep"}, minimum_frames=2)
    assert warnings == []
    assert written == []
    assert not photo.with_suffix(".xmp").exists()
    written, warnings = write_library_sidecars(
        library,
        {photo: "keep"},
        stars={photo: 1},
        minimum_frames=2,
    )
    assert warnings == []
    text = written[0].read_text(encoding="utf-8")
    assert 'xmp:Rating="1"' in text
    assert 'xmpDM:pick="0"' in text


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
