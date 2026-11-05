"""Write a tiny JPEG burst so you can try Outburst without RAW files.

``scene`` draws one synthetic frame. ``main`` writes six JPEGs in two bursts under ``examples/sample_burst``. Run ``python examples/generate_sample_burst.py``, then ``outburst examples/sample_burst``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from outburst.metadata import inject_exif


def scene(width: int, height: int, bird_x: int, sharp: bool, blown: bool) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    sky = np.zeros((height, width, 3), dtype=np.uint8)
    sky[:, :, 0] = 180
    sky[:, :, 1] = 120
    sky[:, :, 2] = 70
    sky[:, :, 0] = np.clip(sky[:, :, 0] - (y * 70 // height), 0, 255)
    ground = y > int(height * 0.72)
    sky[ground, 0] = 50
    sky[ground, 1] = 90
    sky[ground, 2] = 40
    image = sky
    body = ((x - bird_x) ** 2) / 38**2 + ((y - int(height * 0.48)) ** 2) / 18**2 <= 1
    image[body] = (30, 30, 30)
    wing = (np.abs(x - bird_x) < 46) & (np.abs(y - int(height * 0.46)) < 6)
    image[wing] = (20, 20, 20)
    beak = (x > bird_x + 30) & (x < bird_x + 52) & (np.abs(y - int(height * 0.48)) < 5)
    image[beak] = (40, 170, 220)
    # Fine feather texture gives a real Laplacian signal. Blur removes it.
    rng = np.random.default_rng(bird_x)
    noise = rng.integers(-70, 70, size=image.shape[:2], dtype=np.int16)
    feathers = body | wing
    textured = image.astype(np.int16)
    textured[feathers, 0] += noise[feathers]
    textured[feathers, 1] += noise[feathers] // 2
    textured[feathers, 2] += noise[feathers] // 3
    image = np.clip(textured, 0, 255).astype(np.uint8)
    if not sharp:
        image = cv2.GaussianBlur(image, (21, 21), 0)
    if blown:
        white = np.full_like(image, 255)
        image = cv2.addWeighted(image, 0.35, white, 0.65, 0)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sample JPEG burst for Outburst.")
    parser.add_argument("folder", nargs="?", type=Path, default=Path("examples/sample_burst"))
    args = parser.parse_args()
    folder = args.folder
    folder.mkdir(parents=True, exist_ok=True)
    base = datetime(2024, 9, 14, 7, 32, 10)
    frames = [
        ("IMG_1001.jpg", scene(960, 640, 470, sharp=True, blown=False), base),
        ("IMG_1002.jpg", scene(960, 640, 490, sharp=False, blown=False), base + timedelta(milliseconds=100)),
        ("IMG_1003.jpg", scene(960, 640, 510, sharp=True, blown=True), base + timedelta(milliseconds=200)),
        ("IMG_1004.jpg", scene(960, 640, 430, sharp=False, blown=False), base + timedelta(milliseconds=300)),
        ("IMG_1101.jpg", scene(960, 640, 360, sharp=True, blown=False), base + timedelta(seconds=4)),
        ("IMG_1102.jpg", scene(960, 640, 390, sharp=False, blown=False), base + timedelta(seconds=4, milliseconds=120)),
    ]
    for name, image, when in frames:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError(f"Could not encode {name}")
        subsec = f"{when.microsecond // 1000:03d}" if when.microsecond else None
        payload = inject_exif(encoded.tobytes(), when.replace(microsecond=0), subsec, "Outburst Sample")
        (folder / name).write_bytes(payload)
        print(folder / name)
    print(f"Open this folder in Outburst: {folder.resolve()}")


if __name__ == "__main__":
    main()
