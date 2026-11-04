"""Sharpness, exposure, and the composite score."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from outburst.models import Frame, Metrics, Weights
from outburst.scoring import (
    SHARPNESS_SCALE,
    composite_score,
    crop_box,
    effective_sharpness,
    exposure_parts,
    laplacian_variance,
    normalize_sharpness,
    recommended_offset,
    score_bgr,
)


def test_blur_lowers_laplacian_variance() -> None:
    rng = np.random.default_rng(2)
    sharp = rng.integers(0, 255, size=(120, 120, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    assert laplacian_variance(sharp) > laplacian_variance(blurred) * 5


def test_exposure_penalizes_clipped_highlights_and_shadows() -> None:
    rng = np.random.default_rng(1)
    balanced = rng.integers(20, 210, size=(80, 80, 3), dtype=np.uint8)
    blown = balanced.copy()
    blown[:40, :, :] = 255
    dark = np.zeros_like(balanced)
    balanced_score, balanced_highlights, _shadows = exposure_parts(balanced)
    blown_score, blown_highlights, _shadows = exposure_parts(blown)
    dark_score, _highlights, dark_shadows = exposure_parts(dark)
    assert balanced_highlights < 0.02
    assert blown_highlights > 0.4
    assert blown_score < balanced_score
    assert dark_shadows > 0.9
    assert dark_score < balanced_score


def test_normalize_sharpness_is_monotonic() -> None:
    assert normalize_sharpness(0) == 0
    assert normalize_sharpness(SHARPNESS_SCALE) == pytest.approx(0.5)
    assert normalize_sharpness(SHARPNESS_SCALE) < normalize_sharpness(SHARPNESS_SCALE * 3)


def test_composite_uses_the_default_weights() -> None:
    metrics = Metrics(
        sharpness=300,
        center_sharpness=300,
        tenengrad=10,
        exposure=1,
        highlight_clip=0,
        shadow_clip=0,
    )
    assert effective_sharpness(metrics, Weights()) == pytest.approx(300)
    assert composite_score(metrics, Weights()) == pytest.approx(0.625)


def test_subject_sharpness_replaces_the_frame_when_requested() -> None:
    image = np.full((200, 200, 3), 40, dtype=np.uint8)
    rng = np.random.default_rng(0)
    image += rng.integers(0, 5, size=image.shape, dtype=np.uint8)
    patch = np.indices((40, 40)).sum(axis=0)
    checker = ((patch // 2) % 2) * 255
    image[0:40, 0:40] = checker[:, :, None]
    metrics = score_bgr(image, subject_box=(0, 0, 40, 40), subject_label="bird", subject_confidence=0.8)
    assert metrics.subject_sharpness is not None
    assert metrics.subject_sharpness > metrics.center_sharpness * 3
    assert metrics.subject_label == "bird"
    plain = composite_score(metrics, Weights(prefer_subject=False))
    subject = composite_score(metrics, Weights(prefer_subject=True))
    assert subject > plain


def test_crop_box_clips_to_the_image() -> None:
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    cropped = crop_box(image, (-10, -10, 20, 20), margin=0)
    assert cropped.shape[0] <= 50
    assert cropped.shape[1] <= 60
    assert cropped.size > 0


def test_recommended_frame_breaks_ties_by_filename() -> None:
    moment_metrics = Metrics(1, 1, 1, 1, 0, 0)

    class Row:
        def __init__(self, composite: float) -> None:
            self.composite = composite
            self.metrics = moment_metrics

    frames = [
        Frame(Path("b.jpg"), "jpeg", None, "missing", None, 1, 1),
        Frame(Path("a.jpg"), "jpeg", None, "missing", None, 1, 1),
    ]
    by_path = {frames[0].path: Row(0.5), frames[1].path: Row(0.5)}
    assert recommended_offset(frames, by_path) == 1


def test_score_rejects_a_grayscale_array() -> None:
    with pytest.raises(ValueError):
        score_bgr(np.zeros((8, 8), dtype=np.uint8))
