"""Eye finder: a large dark eye is rated, a small one is not, and blur lowers the score.

``_portrait`` draws a synthetic eye. The tests check size, blur, a frame with no eye, a limited search box, and a burst whose sharpest rated eye becomes the keeper.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from outburst.eyes import MIN_RATED_DIAMETER, locate_eye
from outburst.pipeline import analyze_folder
from outburst.scoring import recommended_offset
from tests.helpers import sharp_frame, write_jpeg


def _portrait(diameter: int, *, blur: int = 0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((220, 260, 3), 168, dtype=np.uint8)
    image += rng.integers(0, 10, size=image.shape, dtype=np.uint8)
    center = (150, 100)
    radius = diameter // 2
    cv2.circle(image, center, radius, (25, 22, 20), -1)
    cv2.circle(image, (center[0] - radius // 3, center[1] - radius // 3), max(2, radius // 5), (245, 245, 245), -1)
    if blur:
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    return image


def test_a_large_eye_is_rated_and_a_small_one_is_not() -> None:
    large = locate_eye(_portrait(MIN_RATED_DIAMETER + 12))
    assert large is not None
    assert large.rated
    assert large.sharpness is not None
    assert large.sharpness > 0
    small = locate_eye(_portrait(MIN_RATED_DIAMETER - 6))
    assert small is not None
    assert small.rated is False
    assert small.sharpness is None


def test_a_sharp_eye_outranks_a_blurred_copy() -> None:
    sharp = locate_eye(_portrait(36))
    soft = locate_eye(_portrait(36, blur=21))
    assert sharp is not None and sharp.sharpness is not None
    assert soft is not None and soft.sharpness is not None
    assert sharp.sharpness > soft.sharpness * 2


def test_a_plain_frame_has_no_eye() -> None:
    rng = np.random.default_rng(4)
    image = rng.integers(80, 160, size=(180, 200, 3), dtype=np.uint8)
    assert locate_eye(image) is None


def test_search_can_be_limited_to_a_box() -> None:
    image = _portrait(36)
    eye = locate_eye(image, (120, 70, 60, 60))
    assert eye is not None and eye.rated
    missed = locate_eye(image, (0, 0, 40, 40))
    assert missed is None


def test_a_burst_with_two_rated_eyes_keeps_the_sharpest(tmp_path: Path) -> None:
    folder = tmp_path / "eyes"
    folder.mkdir()
    moment = datetime(2024, 7, 2, 9, 0, 0)
    sharp_path = folder / "sharp.jpg"
    soft_path = folder / "soft.jpg"
    texture_path = folder / "texture.jpg"
    tiny_path = folder / "tiny.jpg"
    write_jpeg(sharp_path, _portrait(36), moment)
    write_jpeg(soft_path, _portrait(36, blur=21), moment + timedelta(milliseconds=200))
    write_jpeg(texture_path, sharp_frame(seed=9), moment + timedelta(milliseconds=400))
    write_jpeg(tiny_path, _portrait(MIN_RATED_DIAMETER - 6), moment + timedelta(milliseconds=600))

    library = analyze_folder(folder, cache_dir=tmp_path / "cache", threshold=1.0, eye_scoring=True)
    assert library.eye_active
    assert len(library.bursts) == 1
    by_path = {item.frame.path: item for item in library.analyses}
    sharp = by_path[sharp_path]
    soft = by_path[soft_path]
    texture = by_path[texture_path]
    tiny = by_path[tiny_path]
    assert sharp.metrics is not None and sharp.metrics.eye_sharpness is not None
    assert soft.metrics is not None and soft.metrics.eye_sharpness is not None
    assert sharp.metrics.eye_sharpness > soft.metrics.eye_sharpness
    assert texture.metrics is not None and texture.metrics.eye_sharpness is None
    assert tiny.metrics is not None and tiny.metrics.eye_radius is not None
    assert tiny.metrics.eye_sharpness is None
    best = library.bursts[0].frames[recommended_offset(library.bursts[0].frames, by_path)]
    assert best.path == sharp_path

    plain = analyze_folder(folder, cache_dir=tmp_path / "cache", threshold=1.0, eye_scoring=False)
    assert plain.eye_active is False
    plain_by_path = {item.frame.path: item for item in plain.analyses}
    assert plain_by_path[sharp_path].metrics is not None
    assert plain_by_path[sharp_path].metrics.eye_sharpness is None
    texture_best = library.bursts[0].frames[recommended_offset(library.bursts[0].frames, plain_by_path)]
    assert texture_best.path == texture_path
