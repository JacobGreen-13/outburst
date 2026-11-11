"""Sharpness and exposure scores computed from preview images.

The Laplacian variance is the primary sharpness signal: higher means more
fine detail. A center crop is blended in because wildlife subjects are often
near the middle of the frame. Exposure penalizes clipped highlights more than
crushed shadows. The composite score is cached as raw measurements plus weights,
so photographers can retune the balance without reading the RAW files again.

``score_bgr`` measures one preview. ``laplacian_variance``, ``tenengrad``, ``exposure_parts``, ``peak_window_sharpness``, and ``composite_score`` are the pieces of that measurement. ``displayed_score`` and ``score_percent`` turn that composite into the 0..100 number on screen. ``recommended_offset`` and ``eye_chooses_best`` pick the best frame. ``compare_partner_offset`` picks the other frame for compare. ``focus_peak_bgr`` paints the strongest edges. ``center_crop`` and ``crop_box`` cut regions. Scores use the preview, not a demosaic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from outburst.models import Frame, Metrics, Weights

# Bump this when the measurement formula changes so cached rows are rebuilt.
SCORE_VERSION = 2
SHARPNESS_SCALE = 300.0
CENTER_CROP_FRACTION = 0.6
SUBJECT_BOX_MARGIN = 0.12
_PEAK_WINDOW_FRACTION = 0.34


class Rankable(Protocol):
    composite: float | None
    metrics: Metrics | None


def score_bgr(
    image: np.ndarray,
    *,
    subject_box: tuple[int, int, int, int] | None = None,
    subject_label: str | None = None,
    subject_confidence: float | None = None,
) -> Metrics:
    """Measure one BGR preview. The subject box is in the same pixel space as `image`.

    Inputs: image, a BGR color preview; subject_box, optional x, y, width, height; subject_label and subject_confidence, optional detector fields.
    Output: a Metrics record with Laplacian, center, Tenengrad, exposure, critical sharpness, and subject sharpness when the box can be measured.
    Uses: the scan pipeline on a preview whose long edge is 1024 px.
    Limitations: the image must be a non-empty BGR array. There is no demosaic. A subject crop whose short side is under 8 px is dropped, and the label is stored only with a measured subject.
    """

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("expected a color image")
    gray = _gray(image)
    sharpness = laplacian_variance(gray)
    center = center_crop(gray, CENTER_CROP_FRACTION)
    edges = tenengrad(gray)
    exposure, highlight_clip, shadow_clip = exposure_parts(gray)
    subject_sharpness = None
    if subject_box is not None:
        region = crop_box(gray, subject_box, SUBJECT_BOX_MARGIN)
        if region.size > 0 and min(region.shape[0], region.shape[1]) >= 8:
            subject_sharpness = peak_window_sharpness(region)
    return Metrics(
        sharpness=sharpness,
        center_sharpness=laplacian_variance(center),
        tenengrad=edges,
        exposure=exposure,
        highlight_clip=highlight_clip,
        shadow_clip=shadow_clip,
        critical_sharpness=peak_window_sharpness(center),
        subject_sharpness=subject_sharpness,
        subject_box=subject_box if subject_sharpness is not None else None,
        subject_label=subject_label if subject_sharpness is not None else None,
        subject_confidence=subject_confidence if subject_sharpness is not None else None,
    )


def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian. Higher means a sharper preview.

    Inputs: image, a BGR preview or a grayscale array.
    Output: the Laplacian variance as a float.
    Uses: full-frame, center, subject, and peak-window sharpness.
    Limitations: the value is measured on the preview, not a demosaiced RAW, and it is not scaled to 0..1.
    """

    gray = _gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(image: np.ndarray) -> float:
    """Mean squared Sobel gradient. Stored as a second opinion, not the default rank.

    Inputs: image, a BGR preview or a grayscale array.
    Output: the mean of the squared 3x3 Sobel gradients.
    Uses: the metrics record and the report. The composite score does not use it.
    Limitations: this does not change which frame is recommended, and it does not demosaic.
    """

    gray = _gray(image)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float((gradient_x * gradient_x + gradient_y * gradient_y).mean())


def exposure_parts(image: np.ndarray) -> tuple[float, float, float]:
    """Return `(score, highlight fraction, shadow fraction)`.

    Highlights at or above 250 and shadows at or below 5 count as clipped.
    A modest amount of specular clipping is tolerated. Large blown areas and
    a frame that is far from a mid tone are penalized. The score is in 0..1.

    Inputs: image, a BGR preview or a grayscale array.
    Output: the exposure score in 0..1, the highlight clip fraction, and the shadow clip fraction.
    Uses: score_bgr. The exposure score is the part the composite blends with sharpness.
    Limitations: clipping is counted on the preview, not a RAW histogram. Highlights are penalized more than shadows.
    """

    gray = _gray(image)
    total = float(gray.size) or 1.0
    highlight_clip = float(np.count_nonzero(gray >= 250)) / total
    shadow_clip = float(np.count_nonzero(gray <= 5)) / total
    highlight_penalty = min(1.0, (highlight_clip / 0.08) ** 1.15)
    shadow_penalty = min(1.0, (shadow_clip / 0.20) ** 1.15)
    mean = float(gray.mean()) / 255.0
    midpoint_penalty = min(1.0, abs(mean - 0.46) / 0.46) * 0.25
    score = max(0.0, 1.0 - 0.75 * highlight_penalty - 0.45 * shadow_penalty - midpoint_penalty)
    return score, highlight_clip, shadow_clip


def normalize_sharpness(value: float) -> float:
    """Map a Laplacian variance onto 0..1 without looking at the rest of the burst.

    Inputs: value, a Laplacian variance.
    Output: value divided by value plus SHARPNESS_SCALE (300), with a negative input treated as zero.
    Uses: composite_score so one frame does not depend on the other frames.
    Limitations: the scale is fixed. A burst-wide rescaling is not applied.
    """

    value = max(0.0, float(value))
    return value / (value + SHARPNESS_SCALE)


def effective_sharpness(metrics: Metrics, weights: Weights) -> float:
    """The sharpness number that actually feeds the composite score.

    With subject scoring on, this is the sharpest patch inside the animal.
    Otherwise the center-weight slider mixes the sharpest patch in the middle
    of the frame with the full-frame measurement.

    Inputs: metrics, the stored measurements; weights, the sharpness, exposure, and center mix, including prefer_subject.
    Output: the Laplacian variance chosen for the composite, before 0..1 normalization.
    Uses: composite_score and the weight sliders, which retune this blend without rereading the photographs.
    Limitations: a missing critical sharpness falls back to center sharpness. Subject sharpness is used only when prefer_subject is set and a subject measurement exists. Eye sharpness is not part of this blend.
    """

    weights = weights.clamped()
    if weights.prefer_subject and metrics.subject_sharpness is not None:
        return float(metrics.subject_sharpness)
    patch = (
        metrics.critical_sharpness
        if metrics.critical_sharpness is not None
        else metrics.center_sharpness
    )
    return float(weights.center_mix * patch + (1.0 - weights.center_mix) * metrics.sharpness)


def peak_window_sharpness(image: np.ndarray) -> float:
    """Laplacian variance of the sharpest patch in `image`.

    Nine windows cover the image. The score is mostly the sharpest window,
    with a little of the next one, so a single bright speck does not decide it.

    Inputs: image, a preview crop such as the center window or the subject box.
    Output: 85 percent of the sharpest window plus 15 percent of the next, or the whole image when it is too small to split.
    Uses: critical sharpness and subject sharpness inside score_bgr.
    Limitations: a short side under 24 px, or a window that covers the crop, returns the whole image. An empty image returns 0.
    """

    if image.size == 0:
        return 0.0
    height, width = image.shape[:2]
    short = min(height, width)
    if short < 24:
        return laplacian_variance(image)
    window = max(16, min(int(round(short * _PEAK_WINDOW_FRACTION)), short))
    if window >= short - 1:
        return laplacian_variance(image)

    def origin(length: int, slot: int) -> int:
        """Place one of the three window slots along an axis.

        Inputs: length, the image height or width; slot, 0 for the start, 1 for the middle, and 2 for the end.
        Output: the pixel origin of that window, never past the end of the axis.
        Uses: peak_window_sharpness when it lays a 3 by 3 grid.
        Limitations: the window size comes from the enclosing function. This does not score the patch.
        """

        if slot <= 0:
            return 0
        if slot == 1:
            return max(0, (length - window) // 2)
        return max(0, length - window)

    samples = [
        laplacian_variance(image[y0 : y0 + window, x0 : x0 + window])
        for row in range(3)
        for col in range(3)
        for y0 in (origin(height, row),)
        for x0 in (origin(width, col),)
    ]
    samples.sort(reverse=True)
    return samples[0] * 0.85 + samples[1] * 0.15


def focus_peak_bgr(image: np.ndarray) -> np.ndarray:
    """Paint the strongest edges green so a loupe can show where detail sits.

    Inputs: image, a BGR color preview.
    Output: a copy of the image with the strongest Laplacian edges painted green.
    Uses: the focus-peak overlay in the preview.
    Limitations: the image must be a non-empty BGR array. Edges at or above the 97th percentile, and at least 12, are painted. The original photograph is not written.
    """

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("expected a color image")
    gray = _gray(image)
    edges = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    cutoff = float(np.percentile(edges, 97))
    mask = edges >= max(cutoff, 12.0)
    painted = image.copy()
    painted[mask] = (80, 255, 120)
    return painted


def composite_score(metrics: Metrics, weights: Weights) -> float:
    """Blend normalized sharpness and exposure. The result is in 0..1.

    Inputs: metrics, the stored measurements; weights, the sharpness and exposure weights.
    Output: the weighted blend of normalized effective sharpness and exposure.
    Uses: the scan, the cache replay, and the weight sliders.
    Limitations: weights are clamped before the blend. Tenengrad and eye sharpness do not enter this number, and the photograph is not read.
    """

    weights = weights.clamped()
    sharpness = normalize_sharpness(effective_sharpness(metrics, weights))
    total = weights.sharpness + weights.exposure
    return (weights.sharpness * sharpness + weights.exposure * metrics.exposure) / total


def displayed_score(composite: float | None) -> int | None:
    """Turn a composite score into the whole number shown in the review window.

    Inputs: composite, a 0..1 score, or None.
    Output: the clamped score times 100, rounded to an integer, or None when composite is None.
    Uses: the filmstrip caption and the minimum-score filter, so both use the same number.
    Limitations: this is the display number. It does not measure the frame again.
    """

    if composite is None:
        return None
    return int(round(max(0.0, min(1.0, composite)) * 100))


def score_percent(composite: float | None) -> str:
    """Show a composite score as a whole number from 0 to 100.

    Inputs: composite, a 0..1 score, or None.
    Output: the display score as text, or an em dash when composite is None.
    Uses: the filmstrip and the score label.
    Limitations: this is display text. It does not measure the frame again.
    """

    score = displayed_score(composite)
    if score is None:
        return "—"
    return str(score)


def _rank_key(frames: Sequence[Frame], by_path: Mapping[Path, Rankable], index: int) -> tuple[float, float, str, str]:
    """Sort key for the composite ranking of one frame in a burst.

    Inputs: frames, the burst in shooting order; by_path, analyses keyed by path; index, the frame to rank.
    Output: a tuple that sorts a higher composite first, then a higher full-frame Laplacian, then the earlier filename.
    Uses: recommended_offset and compare_partner_offset when fewer than two eyes are rated.
    Limitations: a missing analysis sorts as the worst score. Eye sharpness is not part of this key.
    """

    frame = frames[index]
    item = by_path.get(frame.path)
    composite = None if item is None else item.composite
    sharpness = None
    if item is not None and item.metrics is not None:
        sharpness = item.metrics.sharpness
    return (
        -(composite if composite is not None else -1.0),
        -(sharpness if sharpness is not None else -1.0),
        frame.path.name.lower(),
        str(frame.path),
    )


def _rated_eye_sharpness(item: Rankable | None) -> float | None:
    """Return the eye sharpness only when the eye was large enough to rate.

    Inputs: item, a frame analysis, or None.
    Output: the eye sharpness as a float, or None when metrics are missing or eye_sharpness is None.
    Uses: eye_chooses_best and the eye ranking key.
    Limitations: the eye comes from the bird-eye pose model. An eye smaller than 16 px is stored with sharpness None and is not rated here.
    """

    if item is None or item.metrics is None or item.metrics.eye_sharpness is None:
        return None
    return float(item.metrics.eye_sharpness)


def eye_chooses_best(frames: Sequence[Frame], by_path: Mapping[Path, Rankable]) -> bool:
    """True when two or more frames have an eye large enough to rate.

    Inputs: frames, one burst; by_path, analyses keyed by path.
    Output: True when at least two frames have a non-None eye sharpness.
    Uses: the recommended frame and the compare partner.
    Limitations: one rated eye, or none, leaves the composite ranking in place. A small eye whose sharpness is None does not count.
    """

    rated = 0
    for frame in frames:
        if _rated_eye_sharpness(by_path.get(frame.path)) is not None:
            rated += 1
            if rated >= 2:
                return True
    return False


def recommended_offset(frames: Sequence[Frame], by_path: Mapping[Path, Rankable]) -> int:
    """Index of the frame that should be offered as the keeper.

    When two or more frames have a rated eye, the sharpest eye wins and an
    earlier filename breaks a tie. Otherwise the higher composite wins, then
    the higher full-frame Laplacian, then the earlier filename.

    Inputs: frames, one burst in shooting order; by_path, analyses keyed by path.
    Output: the index of the recommended frame. An empty burst returns 0.
    Uses: the review window, the XMP recommended flag, and the report.
    Limitations: this does not write a keeper mark. Eye ranking needs two rated eyes, and an eye under 16 px is left out.
    """

    if not frames:
        return 0
    if eye_chooses_best(frames, by_path):
        return min(range(len(frames)), key=lambda index: _eye_rank_key(frames, by_path, index))
    return min(range(len(frames)), key=lambda index: _rank_key(frames, by_path, index))


def _eye_rank_key(
    frames: Sequence[Frame],
    by_path: Mapping[Path, Rankable],
    index: int,
) -> tuple[float, str, str]:
    """Sort key for the eye ranking of one frame.

    Inputs: frames, the burst; by_path, analyses keyed by path; index, the frame to rank.
    Output: a tuple that sorts a higher rated eye sharpness first, then the earlier filename.
    Uses: recommended_offset and compare_partner_offset when eye_chooses_best is true.
    Limitations: a frame with no rated eye sorts last. An eye under 16 px has sharpness None and does not win.
    """

    frame = frames[index]
    sharpness = _rated_eye_sharpness(by_path.get(frame.path))
    return (
        -(sharpness if sharpness is not None else -1.0),
        frame.path.name.lower(),
        str(frame.path),
    )


def compare_partner_offset(
    frames: Sequence[Frame],
    by_path: Mapping[Path, Rankable],
    current: int,
) -> int | None:
    """The other frame to put beside `current`.

    That is the recommended frame, or the next one in the same ranking when
    `current` is already the recommendation. A one-frame burst has no partner.

    Inputs: frames, one burst; by_path, analyses keyed by path; current, the index on screen.
    Output: the partner index, or None when the burst has fewer than two frames or current is out of range.
    Uses: the compare view.
    Limitations: the partner follows the eye ranking only when two or more eyes are rated. Otherwise it follows the composite ranking.
    """

    if len(frames) < 2 or not 0 <= current < len(frames):
        return None
    rank = _eye_rank_key if eye_chooses_best(frames, by_path) else _rank_key
    order = sorted(range(len(frames)), key=lambda index: rank(frames, by_path, index))
    if current == order[0]:
        return order[1]
    return order[0]


def center_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    """Return the central window. `fraction` is the share of width and of height.

    Inputs: image, a preview array; fraction, the share of width and of height to keep.
    Output: a view of the central crop, at least one pixel on each side.
    Uses: score_bgr for center sharpness and critical sharpness. The usual fraction is 0.6.
    Limitations: fraction is clamped to 0.05..1. This does not copy the pixels or write a file.
    """

    fraction = min(1.0, max(0.05, fraction))
    height, width = image.shape[:2]
    crop_height = max(1, int(round(height * fraction)))
    crop_width = max(1, int(round(width * fraction)))
    y0 = max(0, (height - crop_height) // 2)
    x0 = max(0, (width - crop_width) // 2)
    return image[y0 : y0 + crop_height, x0 : x0 + crop_width]


def crop_box(
    image: np.ndarray,
    box: tuple[int, int, int, int],
    margin: float = SUBJECT_BOX_MARGIN,
) -> np.ndarray:
    """Crop to a bounding box expanded by `margin` and clipped to the image.

    Inputs: image, a preview array; box, x, y, width, height in the same pixel space; margin, the fraction of the box added on each side.
    Output: the cropped view, or an empty array when the expanded box falls outside the image.
    Uses: subject sharpness in score_bgr. The default margin is 0.12.
    Limitations: the box is clipped to the image. An inverted box returns an empty array, and the photograph is not written.
    """

    x, y, width, height = box
    pad_x = int(round(max(0, width) * margin))
    pad_y = int(round(max(0, height) * margin))
    image_height, image_width = image.shape[:2]
    x0 = max(0, int(x) - pad_x)
    y0 = max(0, int(y) - pad_y)
    x1 = min(image_width, int(x) + int(width) + pad_x)
    y1 = min(image_height, int(y) + int(height) + pad_y)
    if x1 <= x0 or y1 <= y0:
        return image[0:0, 0:0]
    return image[y0:y1, x0:x1]


def _gray(image: np.ndarray) -> np.ndarray:
    """Return a single-channel image for the sharpness and exposure measurements.

    Inputs: image, a grayscale array or a BGR color array.
    Output: the same array when it is already two-dimensional, otherwise a grayscale conversion.
    Uses: Laplacian variance, Tenengrad, exposure, and the focus-peak overlay.
    Limitations: a color image is treated as BGR. This does not demosaic a RAW file.
    """

    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
