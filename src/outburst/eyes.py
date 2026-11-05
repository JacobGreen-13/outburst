"""Locate a bird eye with an open-weight pose model and measure its sharpness.

The checkbox is Bird eye detection. It downloads ``eye_pose_v0.pt`` (synthet/eye-pose-v0, YOLO11n-pose trained on CUB-200-2011) and reads the left-eye and right-eye keypoints. A keypoint below 0.35 confidence is ignored. Sharpness is the Laplacian variance inside a disk sized from the bird box, and it is stored only when that disk is at least 16 px across. This is birds only. It does not judge a blink or a catchlight, and it does not look for mammal eyes.

``measure_eye`` scores a disk the caller already located. ``eye_from_pose`` reads one pose result. ``load_bird_eye_detector`` opens weights that are already on disk. ``BirdEyeDetector.find_many`` runs a batch on one thread.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

log = logging.getLogger(__name__)

# On the analysis preview, a smaller disk is too few pixels to trust.
MIN_RATED_DIAMETER = 16
# Keypoint indexes published with synthet/eye-pose-v0.
LEFT_EYE_INDEX = 1
RIGHT_EYE_INDEX = 2
MIN_KEYPOINT_CONFIDENCE = 0.35


@dataclass(frozen=True)
class Eye:
    """An eye in preview pixels. ``sharpness`` is set only when the eye is large enough."""

    x: int
    y: int
    radius: int
    sharpness: float | None

    @property
    def rated(self) -> bool:
        """Whether this eye has a sharpness measurement.

        Inputs: none beyond this eye.
        Output: True when sharpness is set.
        Uses: callers that skip an eye too small to rank.
        Limitations: a detected eye with no sharpness is not rated.
        """

        return self.sharpness is not None


class EyeFinder(Protocol):
    def find_many(self, images: list[np.ndarray]) -> list[Eye | None]:
        """Return one eye, or none, for each preview.

        Inputs: images, BGR previews in order.
        Output: one Eye or None per image.
        Uses: the scan pipeline on the coordinating thread.
        Limitations: the model behind an implementation is not shared across threads.
        """


def measure_eye(image: np.ndarray, x: int, y: int, radius: int) -> Eye | None:
    """Measure sharpness inside a disk that a detector already placed.

    Inputs: image, a BGR preview; x and y, the eye center in preview pixels; radius, the disk radius.
    Output: an Eye. Sharpness is set only when the diameter is at least MIN_RATED_DIAMETER. None when the image is not a color frame or the center is outside it.
    Uses: eye_from_pose, and tests that place an eye without running the pose model.
    Limitations: a disk under 16 px across is returned with sharpness None. A disk with fewer than 40 Laplacian samples stores 0.0. This does not search the image.
    """

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0 or radius < 1:
        return None
    height, width = image.shape[:2]
    center_x = int(x)
    center_y = int(y)
    if not (0 <= center_x < width and 0 <= center_y < height):
        return None
    sharpness = None
    if radius * 2 >= MIN_RATED_DIAMETER:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = _masked_sharpness(gray, center_x, center_y, int(radius))
    return Eye(center_x, center_y, int(radius), sharpness)


def eye_from_pose(result: object, image: np.ndarray) -> Eye | None:
    """Read the best bird eye from one pose prediction.

    Inputs: result, an Ultralytics pose result with keypoints and boxes; image, the BGR preview those coordinates belong to.
    Output: one Eye, or None when no eye keypoint clears MIN_KEYPOINT_CONFIDENCE.
    Uses: BirdEyeDetector.find_many.
    Limitations: only keypoint indexes 1 and 2 (left and right eye of eye-pose-v0) are read. A confident pair keeps the sharper disk when both are large enough, otherwise the higher-confidence eye. The radius is 8% of the shorter side of the bird box, clamped to 4–48 px. A result with no boxes still accepts keypoints and uses a radius of 8.
    """

    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None
    xy = _as_numpy(getattr(keypoints, "xy", None))
    if xy is None or xy.ndim != 3 or xy.shape[0] == 0 or xy.shape[1] <= RIGHT_EYE_INDEX:
        return None
    confidences = _as_numpy(getattr(keypoints, "conf", None))
    boxes = getattr(result, "boxes", None)
    xyxy = None
    box_conf = None
    if boxes is not None and len(boxes) > 0:
        xyxy = _as_numpy(getattr(boxes, "xyxy", None))
        box_conf = _as_numpy(getattr(boxes, "conf", None))
    height, width = image.shape[:2]
    best: tuple[tuple[float, float, float], Eye] | None = None
    for index in range(xy.shape[0]):
        detection_conf = 0.0
        if box_conf is not None and index < len(box_conf):
            detection_conf = float(box_conf[index])
        box = None
        if xyxy is not None and index < len(xyxy):
            x1, y1, x2, y2 = (float(value) for value in xyxy[index][:4])
            box = (x1, y1, x2, y2)
        radius = _eye_radius(box, width, height)
        found: list[tuple[float, Eye]] = []
        for key_index in (LEFT_EYE_INDEX, RIGHT_EYE_INDEX):
            key_conf = 1.0 if confidences is None else float(confidences[index][key_index])
            if key_conf < MIN_KEYPOINT_CONFIDENCE:
                continue
            point_x = float(xy[index][key_index][0])
            point_y = float(xy[index][key_index][1])
            eye = measure_eye(image, int(round(point_x)), int(round(point_y)), radius)
            if eye is not None:
                found.append((key_conf, eye))
        chosen = _choose_eye(found)
        if chosen is None:
            continue
        key_conf, eye = chosen
        rank = (1.0 if eye.sharpness is not None else 0.0, detection_conf, key_conf)
        if best is None or rank > best[0]:
            best = (rank, eye)
    if best is None:
        return None
    return best[1]


def load_bird_eye_detector(model_path: Path, device: str | None = None) -> BirdEyeDetector:
    """Load bird-eye pose weights that are already on disk. This does not download them.

    Inputs: model_path, the .pt file; device, an optional torch device. The subject detector's device order is used when device is omitted.
    Output: a BirdEyeDetector bound to that file and device.
    Uses: analyze_folder when Bird eye detection is on and no finder was passed in.
    Limitations: a missing file raises FileNotFoundError, and a missing ultralytics install raises ImportError. The weights are AGPL-3.0, separate from Outburst, and were trained on CUB-200 bird photos. The returned detector must stay on one thread.
    """

    if not model_path.is_file():
        raise FileNotFoundError(f"Bird eye model not found: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Bird eye detection needs the optional extra: pip install 'outburst[subject]'"
        ) from exc
    from outburst.subject import preferred_device

    chosen = device or preferred_device()
    log.info("Loading bird eye model %s on %s", model_path, chosen)
    return BirdEyeDetector(YOLO(str(model_path)), chosen)


class BirdEyeDetector:
    """Ultralytics pose wrapper that returns one bird eye per preview."""

    def __init__(self, model: object, device: str) -> None:
        """Keep one loaded pose model and the device it runs on.

        Inputs: model, an ultralytics YOLO pose instance; device, `mps`, `cuda`, or `cpu`.
        Output: none. The model and device are stored on the instance.
        Uses: load_bird_eye_detector.
        Limitations: the instance is not shared across threads. Loading it does not download weights.
        """

        self._model = model
        self.device = device

    def find_many(self, images: list[np.ndarray]) -> list[Eye | None]:
        """Predict a batch and return one eye, or none, per preview.

        Inputs: images, BGR previews in batch order.
        Output: one Eye or None per image. An empty input returns an empty list.
        Uses: the scan pipeline, on the coordinating thread.
        Limitations: the model is not shared across threads. Images are converted to RGB and predicted at imgsz 640. A short result list is padded with None. A prediction error returns None for every image in the batch.
        """

        if not images:
            return []
        rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        predict = getattr(self._model, "predict")
        try:
            results = predict(rgb, verbose=False, device=self.device, imgsz=640)
        except Exception:
            log.warning("Bird eye detection failed for %d frame(s)", len(images), exc_info=True)
            return [None] * len(images)
        if not results:
            return [None] * len(images)
        parsed = [eye_from_pose(result, image) for result, image in zip(results, images, strict=False)]
        if len(parsed) < len(images):
            parsed.extend([None] * (len(images) - len(parsed)))
        return parsed[: len(images)]


def _choose_eye(found: list[tuple[float, Eye]]) -> tuple[float, Eye] | None:
    """Pick the sharper rated eye, or the more confident eye when none is rated.

    Inputs: found, confidence and Eye pairs for one bird.
    Output: the chosen pair, or None when found is empty.
    Uses: eye_from_pose.
    Limitations: an unrated eye never outranks a rated one on the same bird.
    """

    if not found:
        return None
    rated = [item for item in found if item[1].sharpness is not None]
    pool = rated or found
    return max(pool, key=lambda item: (item[1].sharpness or 0.0, item[0]))


def _eye_radius(
    box: tuple[float, float, float, float] | None,
    width: int,
    height: int,
) -> int:
    """Estimate an eye radius from the bird box.

    Inputs: box, x1, y1, x2, y2 in preview pixels, or None; width and height of the preview.
    Output: a radius from 4 px up to 48 px, and no more than a quarter of the frame.
    Uses: eye_from_pose. The pose model does not predict an eye radius.
    Limitations: a missing box returns 8. The fraction is 8% of the shorter box side, which is a drawing size, not an anatomical measurement.
    """

    limit = max(4, min(48, width // 4, height // 4))
    if box is None:
        return min(8, limit)
    x1, y1, x2, y2 = box
    span = min(max(1.0, x2 - x1), max(1.0, y2 - y1))
    radius = int(round(span * 0.08))
    return max(4, min(radius, limit))


def _masked_sharpness(gray: np.ndarray, x: int, y: int, radius: int) -> float:
    """Variance of the Laplacian inside the eye disk.

    Inputs: a grayscale image and a disk center and radius.
    Output: a sharpness number, or 0.0 when fewer than 40 pixels are inside the disk.
    Uses: measure_eye when the eye is large enough to rate.
    Limitations: a disk with fewer than 40 Laplacian samples returns 0.0. The mask is clipped at the frame edge.
    """

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(mask, (x, y), radius, 255, -1)
    values = lap[mask > 0]
    if values.size < 40:
        return 0.0
    return float(values.var())


def _as_numpy(value: object) -> np.ndarray | None:
    """Copy a pose tensor onto a NumPy array.

    Inputs: value, a tensor, an array, or None.
    Output: a NumPy array, or None when value is None.
    Uses: eye_from_pose so MPS or CUDA keypoints can be read as pixels.
    Limitations: this moves values to the CPU for reading. It does not run the model.
    """

    if value is None:
        return None
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()
    return np.asarray(value)
