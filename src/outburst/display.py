"""Plain-text and HTML snippets for the review window.

Kept outside the Qt layer so the wording can be tested without a display.

``filmstrip_order``, ``visible_bursts``, and ``visible_frame_indexes`` choose which frames appear and in what order. ``frame_meets_minimum_score`` compares a frame with the minimum score using the number on the filmstrip. ``burst_list_text``, ``score_caption``, ``thumb_badge``, ``frame_info_html``, ``frame_info_parts``, and ``format_captured`` build the labels. ``detected_eye_count`` counts frames with a bird eye. ``obvious_misses`` and ``clip_status`` find the frames G can reject.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from html import escape
from pathlib import Path

from outburst.models import Burst, Frame, FrameAnalysis, Metrics, Weights
from outburst.scoring import displayed_score, effective_sharpness, recommended_offset, score_percent

# A frame this far below the best score is called soft in the filmstrip.
SOFT_SCORE_GAP = 0.15
# Larger gaps and heavy clipping are the misses G can reject.
OBVIOUS_SCORE_GAP = 0.18
OBVIOUS_HIGHLIGHT_CLIP = 0.08
OBVIOUS_SHADOW_CLIP = 0.25


def filmstrip_order(scores: Sequence[float | None], mode: str) -> list[int]:
    """Indexes for the filmstrip. Time keeps shooting order. Score puts the highest first.

    Equal scores stay in the order the frames were shot. A frame with no score
    stays at the right.

    Inputs: one score per frame, which may be None, and a mode string.
    Output: indexes into that sequence.
    Uses: the review window filmstrip.
    Limitations: any mode other than score keeps shooting order.
    """

    order = list(range(len(scores)))
    if mode != "score":
        return order
    return sorted(order, key=lambda index: (scores[index] is None, -(scores[index] or 0.0), index))


def frame_meets_minimum_score(composite: float | None, minimum: int) -> bool:
    """Report whether a frame's on-screen score meets the minimum.

    Inputs: composite, a 0..1 score or None; minimum, the cutoff from 0 to 100.
    Output: True when the number shown on the filmstrip is at least the cutoff.
    Uses: hiding frames and bursts below Minimum score.
    Limitations: a missing score does not meet the cutoff. The cutoff is clamped to 0..100. The comparison uses the rounded display number, so a frame labeled 50 stays visible at a minimum of 50.
    """

    score = displayed_score(composite)
    if score is None:
        return False
    return score >= max(0, min(100, int(minimum)))


def visible_frame_indexes(
    frames: Sequence[Frame],
    by_path: dict[Path, FrameAnalysis],
    *,
    minimum_score: int | None,
) -> list[int]:
    """Indexes of frames that stay on the filmstrip.

    Inputs: frames, one burst in shooting order; by_path, analyses keyed by path; minimum_score, the cutoff, or None when the filter is off.
    Output: shooting-order indexes. None returns every index. A cutoff returns the frames whose display score meets it.
    Uses: the filmstrip and the burst list while Minimum score is on.
    Limitations: a frame with no analysis is hidden while the filter is on. This does not change keeper marks or the files on disk.
    """

    indexes = list(range(len(frames)))
    if minimum_score is None:
        return indexes
    return [
        index
        for index in indexes
        if frame_meets_minimum_score(
            None if (analysis := by_path.get(frames[index].path)) is None else analysis.composite,
            minimum_score,
        )
    ]


def visible_bursts(
    bursts: Sequence[Burst],
    *,
    minimum_frames: int,
    by_path: dict[Path, FrameAnalysis] | None = None,
    minimum_score: int | None = None,
) -> tuple[Burst, ...]:
    """Bursts long enough to list. Shorter bursts stay hidden and are not saved.

    Inputs: bursts; minimum_frames, the shortest burst that stays listed; by_path, analyses keyed by path; minimum_score, the score cutoff, or None when that filter is off.
    Output: the bursts that meet the length and, when a cutoff is set, still have a frame at that score, in the same order.
    Uses: the review window burst list.
    Limitations: a minimum below 1 is treated as 1. A cutoff with no analyses hides every burst. This function does not save files. A burst hidden by the score can still contain a marked keeper.
    """

    minimum = max(1, int(minimum_frames))
    chosen: list[Burst] = []
    for burst in bursts:
        if burst.count < minimum:
            continue
        if minimum_score is not None:
            if by_path is None or not visible_frame_indexes(
                burst.frames, by_path, minimum_score=minimum_score
            ):
                continue
        chosen.append(burst)
    return tuple(chosen)


def human_burst_name(burst_id: str) -> str:
    """Turn a burst id such as burst-003 into Burst 3.

    Inputs: a burst id string.
    Output: a display name, or the original id when it is not burst- plus digits.
    Uses: the burst list and the preview panel.
    Limitations: only the burst- prefix with trailing digits is rewritten.
    """

    prefix = "burst-"
    if burst_id.startswith(prefix) and burst_id[len(prefix) :].isdigit():
        return f"Burst {int(burst_id[len(prefix) :])}"
    return burst_id


def format_captured(moment: datetime | None) -> str:
    """Format a capture time for the burst list and preview panel.

    Inputs: a capture datetime, or None.
    Output: a clock string, or the text No capture time.
    Uses: burst_list_text and frame_info_html.
    Limitations: a missing time is a fixed sentence, and milliseconds appear only when microseconds are set.
    """

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
    *,
    shown_frames: int | None = None,
) -> str:
    """Two lines for one row of the burst list.

    Inputs: a burst, analyses keyed by path, keep or reject marks, and shown_frames, how many frames remain after Minimum score, or None when that filter is off.
    Output: a name, frame count, capture time, best score, and optional eye or keeper text. A shown count below the burst length reads as that many of the total.
    Uses: the review window burst list.
    Limitations: an empty burst is labeled Empty, and a missing analysis shows an em dash for the score. The eye count is every frame with a stored eye center, including an eye too small to rate. A shown count equal to the burst length uses the plain frame count.
    """

    when = format_captured(burst.started_at)
    if not burst.frames:
        return f"{human_burst_name(burst.id)}\nEmpty"
    best = by_path.get(burst.frames[recommended_offset(burst.frames, by_path)].path)
    score = score_percent(None if best is None else best.composite)
    state = burst_state(burst, decisions)
    suffix = f"  ·  {state}" if state else ""
    found = detected_eye_count(
        analysis
        for frame in burst.frames
        if (analysis := by_path.get(frame.path)) is not None
    )
    if found == 1:
        eye = "  ·  1 eye"
    elif found > 1:
        eye = f"  ·  {found} eyes"
    else:
        eye = ""
    if shown_frames is not None and shown_frames != burst.count:
        count = f"{shown_frames} of {burst.count} frames"
    else:
        noun = "frame" if burst.count == 1 else "frames"
        count = f"{burst.count} {noun}"
    return f"{human_burst_name(burst.id)}  ·  {count}\n{when}  ·  best {score}{eye}{suffix}"


def score_caption(
    composite: float | None,
    metrics: Metrics | None,
    *,
    best_composite: float | None,
) -> str:
    """Score for a filmstrip card, with CLIP or SOFT when the frame needs a look.

    Inputs: this frame's score, its metrics, and the burst's best score.
    Output: a short caption such as a percent, EYE, CLIP, or SOFT.
    Uses: the review window filmstrip cards.
    Limitations: a None score or missing metrics omits the matching note. EYE is shown when an eye center was stored, including an eye too small to rate. The star rating stays on the rating buttons.
    """

    text = score_percent(composite)
    notes: list[str] = []
    if metrics is not None and metrics.eye_center is not None and metrics.eye_radius is not None:
        notes.append("EYE")
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
    return text


def detected_eye_count(analyses: Sequence[FrameAnalysis]) -> int:
    """Count frames that have a stored bird-eye center.

    Inputs: analyses, the scored frames.
    Output: how many of those frames have an eye center and radius.
    Uses: the burst list, the status bar, and the hidden-burst message.
    Limitations: an eye too small to rate still counts. A frame with no metrics does not.
    """

    total = 0
    for analysis in analyses:
        metrics = analysis.metrics
        if metrics is not None and metrics.eye_center is not None and metrics.eye_radius is not None:
            total += 1
    return total


def obvious_misses(
    frames: Sequence[Frame],
    by_path: dict[Path, FrameAnalysis],
    decisions: dict[Path, str],
) -> tuple[Path, ...]:
    """Frames a photographer can drop without touching the best one or a keeper.

    A miss is much softer than the recommended frame, or heavily clipped.
    The recommended frame stays, and so does any frame already marked keep.

    Inputs: frames in a burst, analyses keyed by path, and current decisions.
    Output: paths of frames that look like obvious misses.
    Uses: the review window when G rejects the obvious misses.
    Limitations: a burst shorter than two frames returns nothing, and a frame already marked keep or reject is skipped.
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
    """Badge text for one filmstrip thumbnail.

    Inputs: whether the frame is the recommended one, and its keep or reject mark.
    Output: REJECT, BEST, KEEP, a combination, or an empty string.
    Uses: the review window filmstrip.
    Limitations: a reject replaces BEST and KEEP.
    """

    if decision == "reject":
        return "REJECT"
    parts: list[str] = []
    if best:
        parts.append("BEST")
    if decision == "keep":
        parts.append("KEEP")
    return " · ".join(parts)


def burst_state(burst: Burst, decisions: dict[Path, str]) -> str:
    """Summarize a burst as keeper, rejected, or unmarked.

    Inputs: a burst and keep or reject marks keyed by path.
    Output: keeper when any frame is kept, rejected when every frame is rejected, otherwise empty.
    Uses: burst_list_text.
    Limitations: an empty burst returns an empty string, and a mix of rejects and unmarked frames is not called rejected.
    """

    marks = [decisions.get(frame.path) for frame in burst.frames]
    if marks and any(mark == "keep" for mark in marks):
        return "keeper"
    if marks and all(mark == "reject" for mark in marks):
        return "rejected"
    return ""


def _eye_line(metrics: Metrics) -> str:
    """One HTML line about the detected eye.

    Inputs: metrics that may include an eye center, radius, and sharpness.
    Output: an HTML div for no eye, an eye too small to rate, or an eye sharpness number.
    Uses: frame_info_html when bird eye detection is on.
    Limitations: a missing center or radius is reported as no bird eye found.
    """

    if metrics.eye_radius is None or metrics.eye_center is None:
        return '<div style="color:#9a9a9a; margin-top:4px;">No bird eye found</div>'
    if metrics.eye_sharpness is None:
        return '<div style="color:#ffe08a; margin-top:4px; font-weight:600;">Bird eye detected · too small to rate</div>'
    return (
        f'<div style="color:#ffe08a; margin-top:4px; font-weight:600;">'
        f"Bird eye detected · sharpness {metrics.eye_sharpness:.0f}</div>"
    )


def clip_status(metrics: Metrics) -> tuple[str, bool]:
    """A short exposure sentence and whether it should be shown as a warning.

    Inputs: highlight and shadow clip measurements.
    Output: a sentence and True when clipping crosses the warning thresholds.
    Uses: frame_info_html for the exposure line.
    Limitations: light clipping is described without a warning.
    """

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
    """Rich text for the preview panel. Filenames are escaped.

    Inputs: frame identity, analysis, weights, recommendation, decision, stars, and subject or eye flags.
    Output: the summary and the detail joined into one HTML fragment.
    Uses: tests and callers that want the whole panel.
    Limitations: a missing preview is described as unread, and error text is escaped. The joined string matches frame_info_parts with no extra separator.
    """

    summary, detail = frame_info_parts(
        filename=filename,
        frame_number=frame_number,
        frame_count=frame_count,
        burst_id=burst_id,
        captured_at=captured_at,
        camera=camera,
        analysis=analysis,
        weights=weights,
        recommended=recommended,
        decision=decision,
        subject_active=subject_active,
        stars=stars,
        eye_active=eye_active,
    )
    return summary + detail


def frame_info_parts(
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
) -> tuple[str, str]:
    """Split the preview text at the Laplacian number.

    Inputs: frame identity, analysis, weights, recommendation, decision, stars, and subject or eye flags.
    Output: the summary through the score and Laplacian line, then the frame, exposure, and other measurements.
    Uses: the Info button on the review preview.
    Limitations: a missing preview puts the error in the summary and returns an empty detail. Filenames and error text are escaped.
    """

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
        f'<div style="margin-top:4px;"><span style="color:#d0d0d0;">{escape(flag)}</span>'
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
        return title + f'<div style="color:#e0a15a; margin-top:6px;">{escape(problem)}</div>', ""

    metrics = analysis.metrics
    score = score_percent(analysis.composite)
    score_bits = [f"Score {score} / 100"]
    if metrics is not None:
        score_bits.append(f"Laplacian {effective_sharpness(metrics, weights):.0f}")
        sentence, warning = clip_status(metrics)
        color = "#e0a15a" if warning else "#8fd9b5"
        exposure_html = f'<div style="color:{color}; margin-top:4px;">{escape(sentence)}</div>'
    else:
        exposure_html = ""
    score_line = " &nbsp;·&nbsp; ".join(escape(bit) for bit in score_bits)
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
    summary = title + f'<div style="color:#e8e8e8; margin-top:6px;">{score_line}</div>'
    detail = f'<div style="color:#9a9a9a; margin-top:4px;">{meta}</div>' + exposure_html + extra
    return summary, detail

