"""Find a wildlife eye in a preview and measure whether it is sharp.

Eye scoring is optional. A dark, round spot counts as an eye. It is rated only
when the spot is large enough on the preview to measure. Smaller eyes are
reported and left out of the rank.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

# On the 1024 px analysis preview, a smaller disk is too few pixels to trust.
MIN_RATED_DIAMETER = 16
_MIN_DETECTED_DIAMETER = 8
_KERNELS = (15, 23, 31, 45, 61)


@dataclass(frozen=True)
class Eye:
    """An eye in preview pixels. ``sharpness`` is set only when the eye is large enough."""

    x: int
    y: int
    radius: int
    sharpness: float | None

    @property
    def rated(self) -> bool:
        return self.sharpness is not None


def locate_eye(
    image: np.ndarray,
    search_box: tuple[int, int, int, int] | None = None,
) -> Eye | None:
    """Return the strongest eye in ``image``, or None when nothing round and dark is found.

    ``search_box`` limits the search to a subject box. Without one, the whole
    preview is searched. The sharpness value is omitted when the eye is smaller
    than ``MIN_RATED_DIAMETER``.
    """

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    region = _clip_box(search_box, width, height)
    x0, y0, region_width, region_height = region
    if region_width < _MIN_DETECTED_DIAMETER or region_height < _MIN_DETECTED_DIAMETER:
        return None
    crop = gray[y0 : y0 + region_height, x0 : x0 + region_width]
    best: tuple[float, Eye] | None = None
    for kernel in _KERNELS:
        if kernel >= min(region_width, region_height):
            continue
        found = _candidates(crop, kernel)
        for local_x, local_y, radius, strength in found:
            diameter = radius * 2
            if diameter < _MIN_DETECTED_DIAMETER:
                continue
            center_x = x0 + local_x
            center_y = y0 + local_y
            if not _inside(center_x, center_y, radius, width, height):
                continue
            contrast = _contrast(gray, center_x, center_y, radius)
            if contrast < 12:
                continue
            score = contrast * strength
            sharpness = _masked_sharpness(gray, center_x, center_y, radius) if diameter >= MIN_RATED_DIAMETER else None
            eye = Eye(center_x, center_y, radius, sharpness)
            if best is None or score > best[0]:
                best = (score, eye)
    if best is None:
        return None
    return best[1]


def _clip_box(
    box: tuple[int, int, int, int] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    if box is None:
        return (0, 0, width, height)
    x, y, box_width, box_height = box
    x0 = min(max(0, int(x)), width)
    y0 = min(max(0, int(y)), height)
    x1 = min(width, x0 + max(0, int(box_width)))
    y1 = min(height, y0 + max(0, int(box_height)))
    return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))


def _candidates(crop: np.ndarray, kernel: int) -> list[tuple[int, int, int, float]]:
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    blackhat = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, element)
    peak = float(blackhat.max())
    if peak < 8:
        return []
    _, mask = cv2.threshold(blackhat, max(8.0, peak * 0.35), 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[tuple[int, int, int, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 12 or perimeter <= 0:
            continue
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.45:
            continue
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        radius_px = int(round(float(radius)))
        local_x = int(round(center_x))
        local_y = int(round(center_y))
        if radius_px < _MIN_DETECTED_DIAMETER // 2:
            continue
        if not (0 <= local_y < blackhat.shape[0] and 0 <= local_x < blackhat.shape[1]):
            continue
        strength = circularity * float(blackhat[local_y, local_x]) / 255.0
        found.append((local_x, local_y, radius_px, strength))
    return found


def _inside(x: int, y: int, radius: int, width: int, height: int) -> bool:
    return radius <= x < width - radius and radius <= y < height - radius


def _contrast(gray: np.ndarray, x: int, y: int, radius: int) -> float:
    height, width = gray.shape[:2]
    disk = np.zeros(gray.shape, dtype=np.uint8)
    ring = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(disk, (x, y), radius, 255, -1)
    outer = min(radius * 2, min(width, height) // 2)
    cv2.circle(ring, (x, y), max(radius + 2, outer), 255, -1)
    ring = cv2.subtract(ring, disk)
    inside = gray[disk > 0]
    outside = gray[ring > 0]
    if inside.size == 0 or outside.size == 0:
        return 0.0
    return float(outside.mean() - inside.mean())


def _masked_sharpness(gray: np.ndarray, x: int, y: int, radius: int) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, -1)
    values = lap[mask > 0]
    if values.size < 40:
        return 0.0
    return float(values.var())
