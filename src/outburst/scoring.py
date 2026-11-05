"""Sharpness and exposure scores computed from preview images.

The Laplacian variance is the primary sharpness signal: higher means more
fine detail. A center crop is blended in because wildlife subjects are often
near the middle of the frame. Exposure penalizes clipped highlights more than
crushed shadows. The composite score is cached as raw measurements plus weights,
so photographers can retune the balance without reading the RAW files again.
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
    """Measure one BGR preview. The subject box is in the same pixel space as `image`."""

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("expected a color image")
    sharpness = laplacian_variance(image)
    center = laplacian_variance(center_crop(image, CENTER_CROP_FRACTION))
    edges = tenengrad(image)
    exposure, highlight_clip, shadow_clip = exposure_parts(image)
    subject_sharpness = None
    if subject_box is not None:
        region = crop_box(image, subject_box, SUBJECT_BOX_MARGIN)
        if region.size > 0 and min(region.shape[0], region.shape[1]) >= 8:
            subject_sharpness = peak_window_sharpness(region)
    return Metrics(
        sharpness=sharpness,
        center_sharpness=center,
        tenengrad=edges,
        exposure=exposure,
        highlight_clip=highlight_clip,
        shadow_clip=shadow_clip,
        critical_sharpness=peak_window_sharpness(center_crop(image, CENTER_CROP_FRACTION)),
        subject_sharpness=subject_sharpness,
        subject_box=subject_box if subject_sharpness is not None else None,
        subject_label=subject_label if subject_sharpness is not None else None,
        subject_confidence=subject_confidence if subject_sharpness is not None else None,
    )


def laplacian_variance(image: np.ndarray) -> float:
    """Variance of the Laplacian. Higher means a sharper preview."""

    gray = _gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def tenengrad(image: np.ndarray) -> float:
    """Mean squared Sobel gradient. Stored as a second opinion, not the default rank."""

    gray = _gray(image)
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return float((gradient_x * gradient_x + gradient_y * gradient_y).mean())


def exposure_parts(image: np.ndarray) -> tuple[float, float, float]:
    """Return `(score, highlight fraction, shadow fraction)`.

    Highlights at or above 250 and shadows at or below 5 count as clipped.
    A modest amount of specular clipping is tolerated. Large blown areas and
    a frame that is far from a mid tone are penalized. The score is in 0..1.
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
    """Map a Laplacian variance onto 0..1 without looking at the rest of the burst."""

    value = max(0.0, float(value))
    return value / (value + SHARPNESS_SCALE)


def effective_sharpness(metrics: Metrics, weights: Weights) -> float:
    """The sharpness number that actually feeds the composite score.

    With subject scoring on, this is the sharpest patch inside the animal.
    Otherwise the center-weight slider mixes the sharpest patch in the middle
    of the frame with the full-frame measurement.
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
    """Paint the strongest edges green so a loupe can show where detail sits."""

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
    """Blend normalized sharpness and exposure. The result is in 0..1."""

    weights = weights.clamped()
    sharpness = normalize_sharpness(effective_sharpness(metrics, weights))
    total = weights.sharpness + weights.exposure
    return (weights.sharpness * sharpness + weights.exposure * metrics.exposure) / total


def score_percent(composite: float | None) -> str:
    """Show a composite score as a whole number from 0 to 100."""

    if composite is None:
        return "—"
    return str(int(round(max(0.0, min(1.0, composite)) * 100)))


def _rank_key(frames: Sequence[Frame], by_path: Mapping[Path, Rankable], index: int) -> tuple[float, float, str, str]:
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
    if item is None or item.metrics is None or item.metrics.eye_sharpness is None:
        return None
    return float(item.metrics.eye_sharpness)


def eye_chooses_best(frames: Sequence[Frame], by_path: Mapping[Path, Rankable]) -> bool:
    """True when two or more frames have an eye large enough to rate."""

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
    """

    if len(frames) < 2 or not 0 <= current < len(frames):
        return None
    rank = _eye_rank_key if eye_chooses_best(frames, by_path) else _rank_key
    order = sorted(range(len(frames)), key=lambda index: rank(frames, by_path, index))
    if current == order[0]:
        return order[1]
    return order[0]


def center_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    """Return the central window. `fraction` is the share of width and of height."""

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
    """Crop to a bounding box expanded by `margin` and clipped to the image."""

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
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
