"""Plain-text and HTML snippets for the review window.

Kept outside the Qt layer so the wording can be tested without a display.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from html import escape
from pathlib import Path

from outburst.models import Burst, Frame, FrameAnalysis, Metrics, Weights
from outburst.scoring import effective_sharpness, eye_chooses_best, recommended_offset, score_percent

# A frame this far below the best score is called soft in the filmstrip.
SOFT_SCORE_GAP = 0.15
# Larger gaps and heavy clipping are the misses G can reject.
OBVIOUS_SCORE_GAP = 0.18
OBVIOUS_HIGHLIGHT_CLIP = 0.08
OBVIOUS_SHADOW_CLIP = 0.25


def visible_bursts(bursts: Sequence[Burst], *, minimum_frames: int) -> tuple[Burst, ...]:
    """Bursts long enough to list. Shorter bursts stay hidden and are not saved."""

    minimum = max(1, int(minimum_frames))
    return tuple(burst for burst in bursts if burst.count >= minimum)


def human_burst_name(burst_id: str) -> str:
    prefix = "burst-"
    if burst_id.startswith(prefix) and burst_id[len(prefix) :].isdigit():
        return f"Burst {int(burst_id[len(prefix) :])}"
    return burst_id


def format_captured(moment: datetime | None) -> str:
    if moment is None:
        return "No capture time"
    text = moment.strftime("%Y-%m-%d %H:%M:%S")
    if moment.microsecond:
        text += f".{moment.microsecond // 1000:03d}"
    return text


def burst_list_text(
    burst: Burst,
    by_path: dict[Path, FrameAnalysis],
    decisions: dict[Path, str],
) -> str:
    when = format_captured(burst.started_at)
    if not burst.frames:
        return f"{human_burst_name(burst.id)}\nEmpty"
    best = by_path.get(burst.frames[recommended_offset(burst.frames, by_path)].path)
    score = score_percent(None if best is None else best.composite)
    state = burst_state(burst, decisions)
    suffix = f"  ·  {state}" if state else ""
    eye = "  ·  eye" if eye_chooses_best(burst.frames, by_path) else ""
    noun = "frame" if burst.count == 1 else "frames"
    return f"{human_burst_name(burst.id)}  ·  {burst.count} {noun}\n{when}  ·  best {score}{eye}{suffix}"


def star_glyphs(stars: int | None) -> str:
    """Five stars, filled up to `stars`. None and 0 are empty."""

    count = 0 if stars is None else max(0, min(5, int(stars)))
    return "★" * count + "☆" * (5 - count)


def score_caption(
    composite: float | None,
    metrics: Metrics | None,
    *,
    best_composite: float | None,
    stars: int | None = None,
) -> str:
    """Score for a filmstrip card, with CLIP or SOFT when the frame needs a look."""

    text = score_percent(composite)
    notes: list[str] = []
    if metrics is not None and (metrics.highlight_clip >= 0.02 or metrics.shadow_clip >= 0.05):
        notes.append("CLIP")
    if (
        best_composite is not None
        and composite is not None
        and best_composite - composite >= SOFT_SCORE_GAP
    ):
        notes.append("SOFT")
    if notes:
        text = f"{text}  {' '.join(notes)}"
    if stars:
        text = f"{text}  ★{stars}"
    return text


def obvious_misses(
    frames: Sequence[Frame],
    by_path: dict[Path, FrameAnalysis],
    decisions: dict[Path, str],
) -> tuple[Path, ...]:
    """Frames a photographer can drop without touching the best one or a keeper.

    A miss is much softer than the recommended frame, or heavily clipped.
    The recommended frame stays, and so does any frame already marked keep.
    """

    if len(frames) < 2:
        return ()
    best_index = recommended_offset(frames, by_path)
    best = by_path.get(frames[best_index].path)
    best_composite = None if best is None else best.composite
    misses: list[Path] = []
    for index, frame in enumerate(frames):
        if index == best_index or decisions.get(frame.path) in {"keep", "reject"}:
            continue
        analysis = by_path.get(frame.path)
        if analysis is None or analysis.metrics is None or analysis.composite is None:
            continue
        metrics = analysis.metrics
        clipped = (
            metrics.highlight_clip >= OBVIOUS_HIGHLIGHT_CLIP
            or metrics.shadow_clip >= OBVIOUS_SHADOW_CLIP
        )
        soft = best_composite is not None and best_composite - analysis.composite >= OBVIOUS_SCORE_GAP
        if clipped or soft:
            misses.append(frame.path)
    return tuple(misses)


def thumb_badge(best: bool, decision: str | None) -> str:
    if decision == "reject":
        return "REJECT"
    parts: list[str] = []
    if best:
        parts.append("BEST")
    if decision == "keep":
        parts.append("KEEP")
    return " · ".join(parts)


def burst_state(burst: Burst, decisions: dict[Path, str]) -> str:
    marks = [decisions.get(frame.path) for frame in burst.frames]
    if marks and any(mark == "keep" for mark in marks):
        return "keeper"
    if marks and all(mark == "reject" for mark in marks):
        return "rejected"
    return ""


def _eye_line(metrics: Metrics) -> str:
    if metrics.eye_radius is None or metrics.eye_center is None:
        return '<div style="color:#9a9a9a; margin-top:4px;">No eye found</div>'
    if metrics.eye_sharpness is None:
        return '<div style="color:#9a9a9a; margin-top:4px;">Eye is too small to rate</div>'
    return (
        f'<div style="color:#e6c15a; margin-top:4px;">Eye sharpness {metrics.eye_sharpness:.0f}</div>'
    )


def clip_status(metrics: Metrics) -> tuple[str, bool]:
    """A short exposure sentence and whether it should be shown as a warning."""

    notes: list[str] = []
    if metrics.highlight_clip >= 0.005:
        notes.append(f"Highlights clipped {metrics.highlight_clip:.1%}")
    if metrics.shadow_clip >= 0.02:
        notes.append(f"Shadows crushed {metrics.shadow_clip:.1%}")
    if not notes:
        return "Exposure looks balanced", False
    warning = metrics.highlight_clip >= 0.02 or metrics.shadow_clip >= 0.05
    return " · ".join(notes), warning


def frame_info_html(
    *,
    filename: str,
    frame_number: int,
    frame_count: int,
    burst_id: str,
    captured_at: datetime | None,
    camera: str | None,
    analysis: FrameAnalysis | None,
    weights: Weights,
    recommended: bool,
    decision: str | None,
    subject_active: bool,
    stars: int | None = None,
    eye_active: bool = False,
) -> str:
    """Rich text for the preview panel. Filenames are escaped."""

    badges: list[str] = []
    if recommended and decision != "reject":
        badges.append('<span style="color:#7dffc3; font-weight:600;">RECOMMENDED</span>')
    if decision == "keep":
        badges.append(
            '<span style="color:#042016; background-color:#3dbe86; font-weight:700; '
            'padding:1px 8px;">KEEPER</span>'
        )
    elif decision == "reject":
        badges.append(
            '<span style="color:#fff5f5; background-color:#8a3a3a; font-weight:700; '
            'padding:1px 8px;">REJECTED</span>'
        )
    badge_html = (" &nbsp; ".join(badges)) if badges else ""
    if decision == "keep":
        flag = "Pick"
    elif decision == "reject":
        flag = "Reject"
    else:
        flag = "No flag"
    if stars is None and decision == "keep":
        star_note = "sidecar writes 5 stars with this pick"
    elif stars is None and decision == "reject":
        star_note = "kept in this review"
    elif stars is None:
        star_note = "no star rating"
    else:
        star_note = f"{stars} star" if stars == 1 else f"{stars} stars"
    mark_line = (
        f'<div style="color:#e6c15a; margin-top:4px;">{star_glyphs(stars)}'
        f' &nbsp;·&nbsp; <span style="color:#d0d0d0;">{escape(flag)}</span>'
        f' &nbsp;·&nbsp; <span style="color:#9a9a9a;">{escape(star_note)}</span></div>'
    )
    title = (
        f'<div><span style="font-size:16px; font-weight:600;">{escape(filename)}</span>'
        f'&nbsp;&nbsp;{badge_html}</div>'
        + mark_line
    )
    if analysis is None or (analysis.error and analysis.metrics is None):
        problem = "Outburst could not read a preview from this file."
        if analysis is not None and analysis.error:
            problem += " " + analysis.error
        return title + f'<div style="color:#e0a15a; margin-top:6px;">{escape(problem)}</div>'

    metrics = analysis.metrics
    score = score_percent(analysis.composite)
    detail_bits = [f"Score {score} / 100"]
    if metrics is not None:
        detail_bits.append(f"Laplacian {effective_sharpness(metrics, weights):.0f}")
        sentence, warning = clip_status(metrics)
        color = "#e0a15a" if warning else "#8fd9b5"
        exposure_html = f'<div style="color:{color}; margin-top:4px;">{escape(sentence)}</div>'
    else:
        exposure_html = ""
    detail = " &nbsp;·&nbsp; ".join(escape(bit) for bit in detail_bits)
    meta_bits = [
        f"Frame {frame_number} of {frame_count}",
        human_burst_name(burst_id),
        format_captured(captured_at),
    ]
    if camera:
        meta_bits.append(camera)
    meta = " &nbsp;·&nbsp; ".join(escape(bit) for bit in meta_bits)
    extra = ""
    if metrics is not None:
        extra = (
            f'<div style="color:#9a9a9a; margin-top:4px;">Full-frame sharpness {metrics.sharpness:.0f}'
            f" &nbsp;·&nbsp; Center {metrics.center_sharpness:.0f}"
        )
        if metrics.critical_sharpness is not None:
            extra += f" &nbsp;·&nbsp; Sharpest patch {metrics.critical_sharpness:.0f}"
        extra += f" &nbsp;·&nbsp; Edge energy {metrics.tenengrad:.0f}</div>"
        if subject_active:
            if metrics.subject_sharpness is not None:
                label = metrics.subject_label or "subject"
                confidence = ""
                if metrics.subject_confidence is not None:
                    confidence = f" {metrics.subject_confidence:.0%}"
                extra += (
                    f'<div style="color:#8fd9b5; margin-top:4px;">Subject {escape(label)}{escape(confidence)}'
                    f" &nbsp;·&nbsp; sharpest patch {metrics.subject_sharpness:.0f}</div>"
                )
            else:
                extra += (
                    '<div style="color:#9a9a9a; margin-top:4px;">'
                    "No animal detected in this preview. Using frame sharpness."
                    "</div>"
                )
        if eye_active:
            extra += _eye_line(metrics)
    return (
        title
        + f'<div style="color:#e8e8e8; margin-top:6px;">{detail}</div>'
        + f'<div style="color:#9a9a9a; margin-top:4px;">{meta}</div>'
        + exposure_html
        + extra
    )

