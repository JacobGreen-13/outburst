"""Review-window wording.

``test_filmstrip_order_keeps_time_and_ranks_score`` checks Time and Score order. The other tests check hidden short bursts, the minimum-score filter, escaped filenames, badges, score captions, and obvious misses.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from outburst.display import (
    burst_list_text,
    detected_eye_count,
    filmstrip_order,
    frame_info_html,
    frame_info_parts,
    frame_meets_minimum_score,
    obvious_misses,
    score_caption,
    thumb_badge,
    visible_bursts,
    visible_frame_indexes,
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


def test_minimum_score_hides_frames_and_bursts_below_the_cutoff() -> None:
    when = datetime(2024, 6, 1, 12, 0, 0)
    strong = Frame(Path("strong.jpg"), "jpeg", when, "exif", None, 1, 1)
    weak = Frame(Path("weak.jpg"), "jpeg", when, "exif", None, 1, 2)
    low = Frame(Path("low.jpg"), "jpeg", when, "exif", None, 1, 3)
    missing = Frame(Path("missing.jpg"), "jpeg", when, "exif", None, 1, 4)
    metrics = Metrics(10, 10, 1, 0.5, 0.0, 0.0)
    by_path = {
        strong.path: FrameAnalysis(strong, metrics, 0.80, None, None, None),
        weak.path: FrameAnalysis(weak, metrics, 0.494, None, None, None),
        low.path: FrameAnalysis(low, metrics, 0.20, None, None, None),
        missing.path: FrameAnalysis(missing, None, None, None, None, None),
    }
    mixed = Burst("burst-001", (strong, weak))
    soft = Burst("burst-002", (low, missing))
    assert frame_meets_minimum_score(0.80, 50) is True
    assert frame_meets_minimum_score(0.50, 50) is True
    assert frame_meets_minimum_score(0.494, 50) is False
    assert frame_meets_minimum_score(None, 50) is False
    assert visible_frame_indexes(mixed.frames, by_path, minimum_score=None) == [0, 1]
    assert visible_frame_indexes(mixed.frames, by_path, minimum_score=50) == [0]
    assert visible_bursts([mixed, soft], minimum_frames=1, by_path=by_path, minimum_score=50) == (mixed,)
    assert "1 of 2 frames" in burst_list_text(mixed, by_path, {}, shown_frames=1)
    assert "2 frames" in burst_list_text(mixed, by_path, {})


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
    summary, detail = frame_info_parts(
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
    assert summary + detail == html
    assert "Laplacian" in summary
    assert "Burst 4" not in summary
    assert "Burst 4" in detail
    assert "Exposure looks balanced" in detail
    assert "Full-frame sharpness" in detail


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


def test_a_detected_eye_is_named_on_the_burst_and_the_card() -> None:
    path = Path("bird.jpg")
    frame = Frame(path, "jpeg", None, "missing", None, 1, 1)
    metrics = Metrics(1, 1, 1, 1, 0, 0, eye_sharpness=40.0, eye_center=(20, 30), eye_radius=12)
    analysis = FrameAnalysis(frame, metrics, 0.7, None, None, None)
    text = burst_list_text(Burst("burst-003", (frame,)), {path: analysis}, {})
    assert "1 eye" in text
    assert score_caption(0.7, metrics, best_composite=None) == "70  EYE"
    assert detected_eye_count((analysis, FrameAnalysis(frame, Metrics(1, 1, 1, 1, 0, 0), 0.2, None, None, None))) == 1
    html = frame_info_html(
        filename=frame.path.name,
        frame_number=1,
        frame_count=1,
        burst_id="burst-003",
        captured_at=None,
        camera=None,
        analysis=analysis,
        weights=Weights(),
        recommended=False,
        decision=None,
        subject_active=False,
        eye_active=True,
    )
    assert "Bird eye detected" in html
