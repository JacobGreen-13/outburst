"""Plain-text and HTML snippets for the review window.

Kept outside the Qt layer so the wording can be tested without a display.
"""

from __future__ import annotations

from datetime import datetime
from html import escape
from pathlib import Path

from outburst.models import Burst, FrameAnalysis, Metrics, Weights
from outburst.scoring import effective_sharpness, recommended_offset, score_percent


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
    noun = "frame" if burst.count == 1 else "frames"
    return f"{human_burst_name(burst.id)}  ·  {burst.count} {noun}\n{when}  ·  best {score}{suffix}"


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
) -> str:
    """Rich text for the preview panel. Filenames are escaped."""

    badges: list[str] = []
    if recommended and decision != "reject":
        badges.append('<span style="color:#7dffc3; font-weight:600;">RECOMMENDED</span>')
    if decision == "keep":
        badges.append('<span style="color:#7dffc3; font-weight:600;">KEEP</span>')
    elif decision == "reject":
        badges.append('<span style="color:#e06c6c; font-weight:600;">REJECTED</span>')
    badge_html = (" &nbsp; ".join(badges)) if badges else ""
    title = (
        f'<div><span style="font-size:16px; font-weight:600;">{escape(filename)}</span>'
        f'&nbsp;&nbsp;{badge_html}</div>'
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
            f" &nbsp;·&nbsp; Edge energy {metrics.tenengrad:.0f}</div>"
        )
        if subject_active:
            if metrics.subject_sharpness is not None:
                label = metrics.subject_label or "subject"
                confidence = ""
                if metrics.subject_confidence is not None:
                    confidence = f" {metrics.subject_confidence:.0%}"
                extra += (
                    f'<div style="color:#8fd9b5; margin-top:4px;">Subject {escape(label)}{escape(confidence)}'
                    f" &nbsp;·&nbsp; sharpness {metrics.subject_sharpness:.0f}</div>"
                )
            else:
                extra += (
                    '<div style="color:#9a9a9a; margin-top:4px;">'
                    "No animal detected in this preview. Using frame sharpness."
                    "</div>"
                )
    return (
        title
        + f'<div style="color:#e8e8e8; margin-top:6px;">{detail}</div>'
        + f'<div style="color:#9a9a9a; margin-top:4px;">{meta}</div>'
        + exposure_html
        + extra
    )

