"""Sharpness, exposure, and the composite score."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from outburst.models import Frame, Metrics, Weights
from outburst.scoring import (
    SHARPNESS_SCALE,
    compare_partner_offset,
    composite_score,
    crop_box,
    effective_sharpness,
    exposure_parts,
    focus_peak_bgr,
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


def test_two_rated_eyes_outrank_a_sharper_frame() -> None:
    from outburst.scoring import eye_chooses_best

    soft_eye = Metrics(80, 80, 80, 80, 0, 0, eye_sharpness=4.0, eye_center=(20, 20), eye_radius=12)
    sharp_eye = Metrics(8, 8, 8, 8, 0, 0, eye_sharpness=40.0, eye_center=(20, 20), eye_radius=12)
    texture = Metrics(200, 200, 200, 200, 0, 0)

    class Row:
        def __init__(self, metrics: Metrics) -> None:
            self.composite = metrics.sharpness
            self.metrics = metrics

    frames = [
        Frame(Path("soft.jpg"), "jpeg", None, "missing", None, 1, 1),
        Frame(Path("texture.jpg"), "jpeg", None, "missing", None, 1, 1),
        Frame(Path("sharp.jpg"), "jpeg", None, "missing", None, 1, 1),
    ]
    by_path = {
        frames[0].path: Row(soft_eye),
        frames[1].path: Row(texture),
        frames[2].path: Row(sharp_eye),
    }
    assert eye_chooses_best(frames, by_path)
    assert recommended_offset(frames, by_path) == 2
    assert compare_partner_offset(frames, by_path, 2) == 0


def test_one_rated_eye_does_not_change_the_rank() -> None:
    from outburst.scoring import eye_chooses_best

    eyed = Metrics(5, 5, 5, 5, 0, 0, eye_sharpness=90.0, eye_center=(20, 20), eye_radius=12)
    texture = Metrics(200, 200, 200, 200, 0, 0)

    class Row:
        def __init__(self, metrics: Metrics) -> None:
            self.composite = metrics.sharpness
            self.metrics = metrics

    frames = [
        Frame(Path("eye.jpg"), "jpeg", None, "missing", None, 1, 1),
        Frame(Path("texture.jpg"), "jpeg", None, "missing", None, 1, 1),
    ]
    by_path = {frames[0].path: Row(eyed), frames[1].path: Row(texture)}
    assert eye_chooses_best(frames, by_path) is False
    assert recommended_offset(frames, by_path) == 1


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


def test_sharpest_center_patch_outranks_an_even_soft_frame() -> None:
    image = np.full((180, 180, 3), 40, dtype=np.uint8)
    rng = np.random.default_rng(1)
    image += rng.integers(0, 4, size=image.shape, dtype=np.uint8)
    patch = np.indices((28, 28)).sum(axis=0)
    checker = ((patch // 2) % 2) * 255
    image[76:104, 76:104] = checker[:, :, None]
    metrics = score_bgr(image)
    assert metrics.critical_sharpness is not None
    assert metrics.critical_sharpness > metrics.center_sharpness * 2
    weights = Weights()
    averaged = weights.center_mix * metrics.center_sharpness + (1 - weights.center_mix) * metrics.sharpness
    assert effective_sharpness(metrics, weights) > averaged


def test_subject_score_uses_the_sharpest_patch_on_the_animal() -> None:
    image = np.full((160, 160, 3), 30, dtype=np.uint8)
    patch = np.indices((24, 24)).sum(axis=0)
    checker = ((patch // 2) % 2) * 255
    image[20:44, 20:44] = checker[:, :, None]
    metrics = score_bgr(image, subject_box=(16, 16, 40, 40), subject_label="bird", subject_confidence=0.9)
    assert metrics.subject_sharpness is not None
    assert metrics.critical_sharpness is not None
    assert effective_sharpness(metrics, Weights(prefer_subject=True)) == pytest.approx(metrics.subject_sharpness)
    assert metrics.subject_sharpness > metrics.critical_sharpness


def test_focus_peak_paints_edges_on_a_sharp_pattern() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[:, ::4, :] = 255
    peaked = focus_peak_bgr(image)
    assert peaked.shape == image.shape
    green = np.all(peaked == (80, 255, 120), axis=2)
    assert int(green.sum()) > 10


def test_compare_partner_is_the_other_top_frame() -> None:
    class Row:
        def __init__(self, composite: float) -> None:
            self.composite = composite
            self.metrics = Metrics(1, 1, 1, 1, 0, 0)

    frames = [
        Frame(Path("soft.jpg"), "jpeg", None, "missing", None, 1, 1),
        Frame(Path("best.jpg"), "jpeg", None, "missing", None, 1, 1),
        Frame(Path("mid.jpg"), "jpeg", None, "missing", None, 1, 1),
    ]
    by_path = {frames[0].path: Row(0.2), frames[1].path: Row(0.9), frames[2].path: Row(0.5)}
    assert recommended_offset(frames, by_path) == 1
    assert compare_partner_offset(frames, by_path, 1) == 2
    assert compare_partner_offset(frames, by_path, 0) == 1
    assert compare_partner_offset(frames[:1], by_path, 0) is None
