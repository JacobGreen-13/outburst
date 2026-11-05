"""Burst grouping from capture timestamps.

``frame`` builds one timed frame. The tests check gaps under, equal to, and just below the threshold, filename ties, missing clocks, an empty list, a zero threshold, and a rejected negative threshold.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from outburst.grouping import group_frames
from outburst.models import Frame


def frame(name: str, when: datetime | None) -> Frame:
    return Frame(
        path=Path(name),
        kind="raw",
        captured_at=when,
        timestamp_source="exif" if when else "missing",
        camera=None,
        size=1,
        mtime_ns=1,
    )


def test_groups_frames_closer_than_the_threshold() -> None:
    start = datetime(2024, 6, 1, 12, 0, 0)
    frames = [
        frame("c.cr3", start + timedelta(milliseconds=400)),
        frame("a.cr3", start),
        frame("b.cr3", start + timedelta(milliseconds=200)),
    ]
    bursts = group_frames(frames, 1.0)
    assert len(bursts) == 1
    assert [item.path.name for item in bursts[0].frames] == ["a.cr3", "b.cr3", "c.cr3"]
    assert bursts[0].id == "burst-001"


def test_gap_equal_to_the_threshold_starts_a_new_burst() -> None:
    start = datetime(2024, 6, 1, 12, 0, 0)
    frames = [frame("a.nef", start), frame("b.nef", start + timedelta(seconds=1))]
    bursts = group_frames(frames, 1.0)
    assert len(bursts) == 2
    assert [burst.count for burst in bursts] == [1, 1]


def test_gap_just_under_the_threshold_stays_together() -> None:
    start = datetime(2024, 6, 1, 12, 0, 0)
    frames = [
        frame("a.arw", start),
        frame("b.arw", start + timedelta(seconds=1) - timedelta(microseconds=1)),
    ]
    assert len(group_frames(frames, 1.0)) == 1


def test_same_timestamp_orders_by_filename() -> None:
    moment = datetime(2024, 6, 1, 12, 0, 0)
    frames = [frame("b.dng", moment), frame("a.dng", moment)]
    bursts = group_frames(frames, 1.0)
    assert [item.path.name for item in bursts[0].frames] == ["a.dng", "b.dng"]


def test_missing_timestamps_stay_alone_after_timed_bursts() -> None:
    start = datetime(2024, 6, 1, 12, 0, 0)
    frames = [
        frame("z.orf", None),
        frame("m.orf", None),
        frame("a.orf", start),
        frame("b.orf", start + timedelta(milliseconds=100)),
    ]
    original = [item.path.name for item in frames]
    bursts = group_frames(frames, 1.0)
    assert [item.path.name for item in frames] == original
    assert [burst.count for burst in bursts] == [2, 1, 1]
    assert [item.path.name for item in bursts[0].frames] == ["a.orf", "b.orf"]
    assert [item.path.name for item in bursts[1].frames] == ["m.orf"]
    assert [item.path.name for item in bursts[2].frames] == ["z.orf"]


def test_empty_input() -> None:
    assert group_frames([], 1.0) == []


def test_zero_threshold_splits_equal_timestamps() -> None:
    moment = datetime(2024, 6, 1, 12, 0, 0)
    bursts = group_frames([frame("a.raf", moment), frame("b.raf", moment)], 0.0)
    assert len(bursts) == 2


def test_negative_threshold_is_rejected() -> None:
    with pytest.raises(ValueError):
        group_frames([], -0.1)
