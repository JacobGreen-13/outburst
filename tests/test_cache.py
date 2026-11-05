"""SQLite cache round trips.

The tests put measurements, decisions, stars, and eye columns, then read them back. ``test_forget_folder_drops_measurements_and_keeps_decisions`` checks that a folder clear leaves picks. ``test_missing_preview_file_is_a_cache_miss`` checks a deleted thumbnail. ``test_legacy_decisions_without_origin_count_as_user_marks`` covers old rows.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from outburst.cache import Cache
from outburst.models import Metrics


def _metrics() -> Metrics:
    return Metrics(
        sharpness=12.5,
        center_sharpness=20,
        tenengrad=40,
        exposure=0.8,
        highlight_clip=0.01,
        shadow_clip=0.02,
        subject_sharpness=33,
        subject_box=(1, 2, 3, 4),
        subject_label="bird",
        subject_confidence=0.91,
        critical_sharpness=77,
    )


def test_metrics_and_decisions_round_trip(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    photo = tmp_path / "outside.jpg"
    preview = tmp_path / "previews" / "one.jpg"
    thumb = tmp_path / "thumbs" / "one.jpg"
    preview.write_bytes(b"preview")
    thumb.write_bytes(b"thumb")
    moment = datetime(2024, 6, 1, 12, 0, 0, 200000)
    cache.put(
        photo,
        10,
        20,
        "",
        captured_at=moment,
        timestamp_source="exif",
        camera="TestCam",
        metrics=_metrics(),
        preview_path=preview,
        thumb_path=thumb,
    )
    record = cache.get(photo, 10, 20, "")
    assert record is not None
    assert record.captured_at == moment
    assert record.camera == "TestCam"
    assert record.metrics.subject_box == (1, 2, 3, 4)
    assert record.metrics.subject_label == "bird"
    assert record.metrics.critical_sharpness == 77
    assert record.preview_path == preview.resolve()
    assert cache.get(photo, 11, 20, "") is None
    assert cache.get(photo, 10, 20, "other-model") is None

    cache.set_decision(photo, "keep")
    cache.set_stars(photo, 4)
    assert cache.decisions_for([photo])[photo] == "keep"
    assert cache.stars_for([photo])[photo] == 4
    cache.set_stars(photo, None)
    assert photo not in cache.stars_for([photo])
    with pytest.raises(ValueError):
        cache.set_stars(photo, 6)
    assert cache.decision_origins([photo])[photo] == "user"
    cache.set_decision(photo, "keep", origin="auto")
    assert cache.decision_origins([photo])[photo] == "auto"
    cache.set_decision(photo, None)
    assert photo not in cache.decisions_for([photo])
    assert photo not in cache.decision_origins([photo])
    cache.close()


def test_missing_preview_file_is_a_cache_miss(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    photo = Path("/tmp/photo.jpg")
    preview = tmp_path / "previews" / "gone.jpg"
    thumb = tmp_path / "thumbs" / "gone.jpg"
    preview.write_bytes(b"x")
    thumb.write_bytes(b"y")
    cache.put(
        photo,
        1,
        1,
        "",
        captured_at=None,
        timestamp_source="missing",
        camera=None,
        metrics=_metrics(),
        preview_path=preview,
        thumb_path=thumb,
    )
    preview.unlink()
    assert cache.get(photo, 1, 1, "") is None
    cache.close()


def _put(cache: Cache, photo: Path, preview_name: str) -> tuple[Path, Path]:
    preview = cache.directory / "previews" / preview_name
    thumb = cache.directory / "thumbs" / preview_name
    preview.write_bytes(b"preview")
    thumb.write_bytes(b"thumb")
    cache.put(
        photo,
        10,
        20,
        "",
        captured_at=None,
        timestamp_source="missing",
        camera=None,
        metrics=_metrics(),
        preview_path=preview,
        thumb_path=thumb,
    )
    return preview, thumb


def test_forget_folder_drops_measurements_and_keeps_decisions(tmp_path: Path) -> None:
    cache = Cache(tmp_path)
    folder = tmp_path / "photos"
    folder.mkdir()
    inside = folder / "bird.jpg"
    outside = tmp_path / "photos-extra" / "other.jpg"
    outside.parent.mkdir()
    inside_preview, inside_thumb = _put(cache, inside, "inside.jpg")
    outside_preview, _outside_thumb = _put(cache, outside, "outside.jpg")
    cache.set_decision(inside, "keep")

    assert cache.forget_folder(folder) == 1
    assert cache.get(inside, 10, 20, "") is None
    assert not inside_preview.exists()
    assert not inside_thumb.exists()
    assert cache.get(outside, 10, 20, "") is not None
    assert outside_preview.exists()
    assert cache.decisions_for([inside])[inside] == "keep"
    assert cache.forget_folder(folder) == 0

    orphan = cache.directory / "previews" / "orphan.jpg"
    orphan.write_bytes(b"orphan")
    assert cache.clear() == 1
    assert cache.get(outside, 10, 20, "") is None
    assert not outside_preview.exists()
    assert not orphan.exists()
    assert cache.decisions_for([inside])[inside] == "keep"
    cache.close()


def test_eye_measurements_round_trip_and_migrate(tmp_path: Path) -> None:
    from dataclasses import replace

    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "previews").mkdir()
    (legacy / "thumbs").mkdir()
    connection = sqlite3.connect(legacy / "outburst.sqlite")
    connection.executescript(
        """
        CREATE TABLE frame_metrics (
            path TEXT NOT NULL,
            mtime_ns INTEGER NOT NULL,
            size INTEGER NOT NULL,
            score_version INTEGER NOT NULL,
            subject_key TEXT NOT NULL,
            captured_at TEXT,
            timestamp_source TEXT,
            camera TEXT,
            sharpness REAL,
            center_sharpness REAL,
            critical_sharpness REAL,
            tenengrad REAL,
            exposure REAL,
            highlight_clip REAL,
            shadow_clip REAL,
            subject_sharpness REAL,
            subject_confidence REAL,
            subject_label TEXT,
            subject_box TEXT,
            preview_rel TEXT,
            thumb_rel TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (path, mtime_ns, size, score_version, subject_key)
        );
        CREATE TABLE decisions (
            path TEXT PRIMARY KEY,
            decision TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            origin TEXT NOT NULL DEFAULT 'user'
        );
        CREATE TABLE ratings (
            path TEXT PRIMARY KEY,
            stars INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    connection.commit()
    connection.close()

    cache = Cache(legacy)
    photo = tmp_path / "eye.jpg"
    preview = legacy / "previews" / "eye.jpg"
    thumb = legacy / "thumbs" / "eye.jpg"
    preview.write_bytes(b"preview")
    thumb.write_bytes(b"thumb")
    cache.put(
        photo,
        10,
        20,
        "eye",
        captured_at=None,
        timestamp_source="missing",
        camera=None,
        metrics=replace(_metrics(), eye_sharpness=18.5, eye_center=(40, 50), eye_radius=11),
        preview_path=preview,
        thumb_path=thumb,
    )
    record = cache.get(photo, 10, 20, "eye")
    assert record is not None
    assert record.metrics.eye_sharpness == 18.5
    assert record.metrics.eye_center == (40, 50)
    assert record.metrics.eye_radius == 11
    cache.close()


def test_legacy_decisions_without_origin_count_as_user_marks(tmp_path: Path) -> None:
    photo = tmp_path / "outside.jpg"
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    connection = sqlite3.connect(legacy / "outburst.sqlite")
    connection.execute(
        """
        CREATE TABLE decisions (
            path TEXT PRIMARY KEY,
            decision TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO decisions (path, decision, updated_at) VALUES (?, 'keep', '2020-01-01T00:00:00+00:00')",
        (str(photo),),
    )
    connection.commit()
    connection.close()

    cache = Cache(legacy)
    assert cache.decisions_for([photo])[photo] == "keep"
    assert cache.decision_origins([photo])[photo] == "user"
    cache.set_decision(photo, "reject", origin="auto")
    assert cache.decisions_for([photo])[photo] == "reject"
    assert cache.decision_origins([photo])[photo] == "user"
    with pytest.raises(ValueError, match="origin"):
        cache.set_decision(photo, "keep", origin="guess")
    cache.close()

    reopened = Cache(legacy)
    assert reopened.decision_origins([photo])[photo] == "user"
    reopened.close()
