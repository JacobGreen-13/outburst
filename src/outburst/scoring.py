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
SCORE_VERSION = 1
SHARPNESS_SCALE = 300.0
CENTER_CROP_FRACTION = 0.6
SUBJECT_BOX_MARGIN = 0.12


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
            subject_sharpness = laplacian_variance(region)
    return Metrics(
        sharpness=sharpness,
        center_sharpness=center,
        tenengrad=edges,
        exposure=exposure,
        highlight_clip=highlight_clip,
        shadow_clip=shadow_clip,
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
    """The sharpness number that actually feeds the composite score."""

    weights = weights.clamped()
    if weights.prefer_subject and metrics.subject_sharpness is not None:
        return float(metrics.subject_sharpness)
    return float(
        weights.center_mix * metrics.center_sharpness
        + (1.0 - weights.center_mix) * metrics.sharpness
    )


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


def recommended_offset(frames: Sequence[Frame], by_path: Mapping[Path, Rankable]) -> int:
    """Index of the frame that should be offered as the keeper.

    Higher composite wins. Ties prefer a higher full-frame Laplacian, then
    the earlier filename, so the choice stays put when two scores match.
    """

    if not frames:
        return 0

    def sort_key(index: int) -> tuple[float, float, str, str]:
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

    return min(range(len(frames)), key=sort_key)


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
