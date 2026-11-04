"""Synthetic photographs for pipeline tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from outburst.metadata import inject_exif


def write_jpeg(
    path: Path,
    image: np.ndarray,
    when: datetime,
    *,
    model: str = "TestCam",
) -> None:
    subsec = None
    if when.microsecond:
        subsec = f"{when.microsecond // 1000:03d}"
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("could not encode a test JPEG")
    path.write_bytes(inject_exif(encoded.tobytes(), when.replace(microsecond=0), subsec, model))


def sharp_frame(seed: int = 2, height: int = 180, width: int = 240) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = rng.integers(30, 200, size=(height, width, 3), dtype=np.uint8)
    image[::6, :, :] = 15
    image[:, ::8, :] = 235
    return image


def blurry_frame(seed: int = 2) -> np.ndarray:
    return cv2.GaussianBlur(sharp_frame(seed), (31, 31), 0)


def clipped_frame(seed: int = 2) -> np.ndarray:
    return np.full_like(sharp_frame(seed), 255)


def write_burst_folder(folder: Path) -> dict[str, Path]:
    """Three frames in one burst and two frames in the next."""

    folder.mkdir(parents=True, exist_ok=True)
    base = datetime(2024, 6, 1, 12, 0, 0)
    specs = [
        ("IMG_001.jpg", sharp_frame(), base),
        ("IMG_002.jpg", blurry_frame(), base + timedelta(milliseconds=200)),
        ("IMG_003.jpg", clipped_frame(), base + timedelta(milliseconds=400)),
        ("IMG_004.jpg", blurry_frame(seed=5), base + timedelta(seconds=5)),
        ("IMG_005.jpg", sharp_frame(seed=5), base + timedelta(seconds=5, milliseconds=200)),
    ]
    written: dict[str, Path] = {"folder": folder}
    keys = ["sharp", "blur", "clipped", "second_soft", "second_sharp"]
    for key, (name, image, when) in zip(keys, specs, strict=True):
        path = folder / name
        write_jpeg(path, image, when)
        written[key] = path
    return written
