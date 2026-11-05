"""Review-window wording."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from outburst.display import (
    burst_list_text,
    filmstrip_order,
    frame_info_html,
    obvious_misses,
    score_caption,
    thumb_badge,
    visible_bursts,
)
from outburst.models import Burst, Frame, FrameAnalysis, Metrics, Weights


def test_filmstrip_order_keeps_time_and_ranks_score() -> None:
    scores = [0.2, 0.9, None, 0.9]
    assert filmstrip_order(scores, "time") == [0, 1, 2, 3]
    assert filmstrip_order(scores, "score") == [1, 3, 0, 2]
    assert filmstrip_order(scores, "other") == [0, 1, 2, 3]


def test_bursts_shorter_than_the_minimum_stay_hidden() -> None:
    when = datetime(2024, 6, 1, 12, 0, 0)
    pair = Burst(
        "burst-001",
        (
            Frame(Path("a.jpg"), "jpeg", when, "exif", None, 1, 1),
            Frame(Path("b.jpg"), "jpeg", when, "exif", None, 1, 1),
        ),
    )
    single = Burst("burst-002", (Frame(Path("c.jpg"), "jpeg", when, "exif", None, 1, 1),))
    assert visible_bursts([pair, single], minimum_frames=2) == (pair,)
    assert visible_bursts([pair, single], minimum_frames=1) == (pair, single)
    assert visible_bursts([pair, single], minimum_frames=3) == ()


def test_html_escapes_filenames_and_marks_the_recommendation() -> None:
    path = Path("a<b>.jpg")
    frame = Frame(path, "jpeg", datetime(2024, 6, 1, 8, 0, 0, 200000), "exif", "Cam", 1, 1)
    metrics = Metrics(10, 12, 3, 0.9, 0.0, 0.0)
    analysis = FrameAnalysis(frame, metrics, 0.8, None, None, None)
    html = frame_info_html(
        filename=frame.path.name,
        frame_number=1,
        frame_count=1,
        burst_id="burst-004",
        captured_at=frame.captured_at,
        camera=frame.camera,
        analysis=analysis,
        weights=Weights(),
        recommended=True,
        decision=None,
        subject_active=False,
    )
    assert "<script>" not in html
    assert "&lt;b&gt;" in html
    assert "RECOMMENDED" in html
    assert "KEEPER" not in html
    kept = frame_info_html(
        filename=frame.path.name,
        frame_number=1,
        frame_count=1,
        burst_id="burst-004",
        captured_at=frame.captured_at,
        camera=frame.camera,
        analysis=analysis,
        weights=Weights(),
        recommended=True,
        decision="keep",
        subject_active=False,
    )
    assert "KEEPER" in kept
    assert "Pick" in kept
    rated = frame_info_html(
        filename=frame.path.name,
        frame_number=1,
        frame_count=1,
        burst_id="burst-004",
        captured_at=frame.captured_at,
        camera=frame.camera,
        analysis=analysis,
        weights=Weights(),
        recommended=False,
        decision=None,
        subject_active=False,
        stars=2,
    )
    assert "★★☆☆☆" in rated
    assert "2 stars" in rated
    assert "No flag" in rated
    assert "Burst 4" in html
    assert "08:00:00.200" in html
    assert "Sharpest patch" not in html


def test_burst_label_and_badge() -> None:
    path = Path("IMG.jpg")
    frame = Frame(path, "jpeg", None, "missing", None, 1, 1)
    analysis = FrameAnalysis(frame, Metrics(1, 1, 1, 1, 0, 0), 0.5, None, None, None)
    burst = Burst("burst-002", (frame,))
    text = burst_list_text(burst, {path: analysis}, {path: "keep"})
    assert "Burst 2" in text
    assert "keeper" in text
    assert "best 50" in text
    assert thumb_badge(True, "keep") == "BEST · KEEP"
    assert thumb_badge(True, "reject") == "REJECT"
    sharp_patch = frame_info_html(
        filename=frame.path.name,
        frame_number=1,
        frame_count=1,
        burst_id="burst-002",
        captured_at=None,
        camera=None,
        analysis=FrameAnalysis(frame, Metrics(1, 1, 1, 1, 0, 0, critical_sharpness=40), 0.5, None, None, None),
        weights=Weights(),
        recommended=False,
        decision=None,
        subject_active=False,
    )
    assert "Sharpest patch 40" in sharp_patch


def test_score_caption_and_obvious_misses() -> None:
    assert score_caption(0.84, Metrics(1, 1, 1, 1, 0, 0), best_composite=0.9) == "84"
    clipped = Metrics(1, 1, 1, 0.2, 0.1, 0)
    assert score_caption(0.4, clipped, best_composite=0.9) == "40  CLIP SOFT"
    best = Frame(Path("best.jpg"), "jpeg", None, "missing", None, 1, 1)
    soft = Frame(Path("soft.jpg"), "jpeg", None, "missing", None, 1, 1)
    kept = Frame(Path("kept.jpg"), "jpeg", None, "missing", None, 1, 1)
    burst_frames = (best, soft, kept)

    def row(path: Path, composite: float, metrics: Metrics) -> FrameAnalysis:
        frame = next(item for item in burst_frames if item.path == path)
        return FrameAnalysis(frame, metrics, composite, None, None, None)

    by_path = {
        best.path: row(best.path, 0.9, Metrics(1, 1, 1, 1, 0, 0)),
        soft.path: row(soft.path, 0.4, Metrics(1, 1, 1, 0.2, 0.2, 0)),
        kept.path: row(kept.path, 0.4, Metrics(1, 1, 1, 0.2, 0.2, 0)),
    }
    misses = obvious_misses(burst_frames, by_path, {kept.path: "keep"})
    assert misses == (soft.path,)
