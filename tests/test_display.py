"""Review-window wording."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from outburst.display import burst_list_text, frame_info_html, thumb_badge, visible_bursts
from outburst.models import Burst, Frame, FrameAnalysis, Metrics, Weights


def test_single_frame_bursts_are_hidden_until_requested() -> None:
    when = datetime(2024, 6, 1, 12, 0, 0)
    pair = Burst(
        "burst-001",
        (
            Frame(Path("a.jpg"), "jpeg", when, "exif", None, 1, 1),
            Frame(Path("b.jpg"), "jpeg", when, "exif", None, 1, 1),
        ),
    )
    single = Burst("burst-002", (Frame(Path("c.jpg"), "jpeg", when, "exif", None, 1, 1),))
    assert visible_bursts([pair, single], show_singles=False) == (pair,)
    assert visible_bursts([pair, single], show_singles=True) == (pair, single)


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
    assert "Burst 4" in html
    assert "08:00:00.200" in html


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
