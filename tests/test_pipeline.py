"""End-to-end scoring on synthetic bursts. Original bytes stay put.

``_digest`` hashes a file. The tests check grouping, the cache, a threshold change that does not reread files, a corrupt RAW, preview threads, subject detection on the caller thread, cancel, and scoring with no detector.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import outburst.preview as preview_module
from outburst.pipeline import analyze_folder, with_threshold, with_weights
from outburst.models import Weights
from outburst.scoring import recommended_offset
from outburst.subject import Detection
from tests.helpers import write_burst_folder, write_jpeg, sharp_frame
from datetime import datetime


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_analyze_groups_and_recommends_the_sharper_frame(tmp_path: Path) -> None:
    written = write_burst_folder(tmp_path / "burst")
    before = {path: _digest(path) for path in written.values() if path.is_file()}
    library = analyze_folder(written["folder"], cache_dir=tmp_path / "cache", threshold=1.0)
    assert [path.stat().st_mtime_ns for path in before]  # files still exist
    assert {path: _digest(path) for path in before} == before
    assert len(library.bursts) == 2
    assert [burst.count for burst in library.bursts] == [3, 2]
    by_path = {item.frame.path: item for item in library.analyses}
    first = library.bursts[0]
    best = first.frames[recommended_offset(first.frames, by_path)]
    assert best.path == written["sharp"]
    assert by_path[written["sharp"]].frame.camera == "TestCam"
    assert by_path[written["sharp"]].frame.captured_at is not None
    assert by_path[written["sharp"]].composite is not None
    assert by_path[written["blur"]].composite is not None
    assert by_path[written["sharp"]].composite > by_path[written["blur"]].composite
    assert by_path[written["sharp"]].composite > by_path[written["clipped"]].composite
    second = library.bursts[1]
    second_best = second.frames[recommended_offset(second.frames, by_path)]
    assert second_best.path == written["second_sharp"]


def test_second_scan_uses_the_cache(tmp_path: Path, monkeypatch) -> None:
    written = write_burst_folder(tmp_path / "burst")
    cache_dir = tmp_path / "cache"
    analyze_folder(written["folder"], cache_dir=cache_dir)
    calls = {"count": 0}
    original = preview_module.extract_preview_bytes

    def counting(path, paired_jpeg=None):  # type: ignore[no-untyped-def]
        calls["count"] += 1
        return original(path, paired_jpeg)

    monkeypatch.setattr(preview_module, "extract_preview_bytes", counting)
    # analyze_folder imports the function into its module globals.
    import outburst.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "extract_preview_bytes", counting)
    again = analyze_folder(written["folder"], cache_dir=cache_dir, weights=Weights(sharpness=0.2, exposure=0.8))
    assert calls["count"] == 0
    assert again.analyses[0].composite is not None


def test_threshold_change_does_not_reread_files(tmp_path: Path) -> None:
    written = write_burst_folder(tmp_path / "burst")
    library = analyze_folder(written["folder"], cache_dir=tmp_path / "cache", threshold=1.0)
    split = with_threshold(library, 0.1)
    assert len(split.bursts) > len(library.bursts)
    rescored = with_weights(library, Weights(sharpness=1.0, exposure=0.0))
    assert rescored.weights.sharpness == 1.0
    assert rescored.analyses[0].metrics == library.analyses[0].metrics


def test_corrupt_raw_is_reported_without_modifying_it(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("outburst.metadata.shutil.which", lambda _name: None)
    raw = tmp_path / "card" / "bird.CR3"
    raw.parent.mkdir()
    raw.write_bytes(b"ftyp" + b"2024:08:09 10:11:12" + b"no-preview")
    before = _digest(raw)
    library = analyze_folder(raw.parent, cache_dir=tmp_path / "cache")
    assert _digest(raw) == before
    assert len(library.analyses) == 1
    assert library.analyses[0].error
    assert library.analyses[0].frame.captured_at == datetime(2024, 8, 9, 10, 11, 12)
    notes = " ".join(library.warnings)
    assert "exifread" in notes
    assert "CR3" in notes


def test_preview_scoring_runs_on_more_than_one_thread(tmp_path: Path, monkeypatch) -> None:
    import threading

    import outburst.pipeline as pipeline_module

    folder = tmp_path / "many"
    folder.mkdir()
    moment = datetime(2024, 3, 1, 8, 0, 0)
    for index in range(4):
        write_jpeg(folder / f"IMG_{index}.jpg", sharp_frame(seed=index), moment)
    idents: set[int] = set()
    original = pipeline_module.score_bgr

    def counting(*args, **kwargs):  # type: ignore[no-untyped-def]
        idents.add(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline_module, "score_bgr", counting)
    library = analyze_folder(folder, cache_dir=tmp_path / "cache", threshold=1.0)
    assert len(library.analyses) == 4
    assert len(idents) >= 2


def test_subject_detection_stays_on_the_calling_thread(tmp_path: Path) -> None:
    import threading

    folder = tmp_path / "birds"
    folder.mkdir()
    moment = datetime(2024, 3, 1, 8, 0, 0)
    for index in range(5):
        write_jpeg(folder / f"IMG_{index}.jpg", sharp_frame(seed=index + 3), moment)
    idents: set[int] = set()

    class FakeDetector:
        def detect(self, image):  # type: ignore[no-untyped-def]
            idents.add(threading.get_ident())
            height, width = image.shape[:2]
            return Detection((0, 0, width // 2, height // 2), "bird", 0.5)

    library = analyze_folder(folder, cache_dir=tmp_path / "cache", detector=FakeDetector(), threshold=1.0)
    assert len(library.analyses) == 5
    assert idents == {threading.get_ident()}
    assert all(item.metrics is not None and item.metrics.subject_label == "bird" for item in library.analyses)


def test_a_failed_detector_names_the_reason(tmp_path: Path) -> None:
    folder = tmp_path / "birds"
    folder.mkdir()
    write_jpeg(folder / "bird.jpg", sharp_frame(), datetime(2024, 3, 1, 8, 0, 0))

    class BrokenDetector:
        def detect_many(self, images):  # type: ignore[no-untyped-def]
            raise RuntimeError("No module named outburst")

    library = analyze_folder(
        folder,
        cache_dir=tmp_path / "cache",
        detector=BrokenDetector(),
        weights=Weights(prefer_subject=True),
    )
    assert any("No module named outburst" in note for note in library.warnings)
    assert library.analyses[0].metrics is not None
    assert library.analyses[0].metrics.subject_label is None


def test_cancel_stops_before_a_library_is_returned(tmp_path: Path) -> None:
    from outburst.pipeline import CancelledError

    folder = tmp_path / "cancel"
    folder.mkdir()
    moment = datetime(2024, 3, 1, 8, 0, 0)
    for index in range(3):
        write_jpeg(folder / f"IMG_{index}.jpg", sharp_frame(seed=index), moment)
    with pytest.raises(CancelledError):
        analyze_folder(folder, cache_dir=tmp_path / "cache", should_cancel=lambda: True)


def test_subject_detector_is_optional(tmp_path: Path) -> None:
    folder = tmp_path / "one"
    folder.mkdir()
    write_jpeg(folder / "bird.jpg", sharp_frame(), datetime(2024, 1, 1, 1, 1, 1))

    class FakeDetector:
        def detect(self, image):  # type: ignore[no-untyped-def]
            height, width = image.shape[:2]
            return Detection((width // 4, height // 4, width // 2, height // 2), "bird", 0.88)

    library = analyze_folder(
        folder,
        cache_dir=tmp_path / "cache",
        detector=FakeDetector(),
        weights=Weights(prefer_subject=True),
    )
    metrics = library.analyses[0].metrics
    assert library.subject_active
    assert metrics is not None
    assert metrics.subject_label == "bird"
    assert metrics.subject_box is not None
    assert metrics.subject_sharpness is not None
