"""Locate a bird eye with an open-weight pose model and measure its sharpness.

The checkbox is Bird eye detection. The install button downloads ``eye_pose_v0.pt`` (synthet/eye-pose-v0, YOLO11n-pose trained on CUB-200-2011). This module reads the left-eye and right-eye keypoints. A keypoint below 0.35 confidence is ignored. Sharpness is the Laplacian variance inside a disk sized from the bird box, and it is stored only when that disk is at least 16 px across. This is birds only. It does not judge a blink or a catchlight, and it does not look for mammal eyes.

``measure_eye`` scores a disk the caller already located. ``eye_from_pose`` reads one pose result. ``pose_device`` keeps this model off MPS. ``load_bird_eye_detector`` opens weights that are already on disk. ``BirdEyeDetector.find_many`` runs a batch on one thread: the whole preview first, then overlapping regions when that preview has no eye.
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
# A bird that is clear but small in a 1024 px preview is missed at imgsz 640.
# Overlapping windows of this size make that bird a larger part of the frame.
TILE_EDGE = 512
# Appended to the weights cache key so an older scan that stored no eye is not reused.
EYE_CACHE_TOKEN = "eye-v2"


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


def pose_device(requested: str | None = None) -> str:
    """Choose the torch device for the bird-eye pose model.

    Inputs: requested, `mps`, `cuda`, `cpu`, or None. None uses the subject detector's device order.
    Output: `cuda` or `cpu`. An MPS request is returned as cpu.
    Uses: load_bird_eye_detector.
    Limitations: this does not load weights. Pose coordinates from MPS have come back unusable, so this model stays on the CPU even when a subject model uses MPS. A missing torch install yields cpu through the subject device order.
    """

    from outburst.subject import preferred_device

    chosen = requested or preferred_device()
    if chosen == "mps":
        return "cpu"
    return chosen


def load_bird_eye_detector(model_path: Path, device: str | None = None) -> EyeFinder:
    """Load bird-eye pose weights that are already on disk. This does not download them.

    Inputs: model_path, the .pt file; device, an optional torch device. The subject detector's device order is used when device is omitted, except that MPS becomes cpu.
    Output: a BirdEyeDetector bound to that file and device.
    Uses: analyze_folder when Bird eye detection is on and no finder was passed in.
    Limitations: a missing file raises FileNotFoundError, and a missing ultralytics install raises ImportError. The weights are AGPL-3.0, separate from Outburst, and were trained on CUB-200 bird photos. The returned detector must stay on one thread. It does not run on MPS.
    """

    if not model_path.is_file():
        raise FileNotFoundError(f"Bird eye model not found: {model_path}")
    from outburst.subject import uses_private_python

    if uses_private_python():
        from outburst.sidecar import SidecarEyeFinder

        return SidecarEyeFinder(model_path)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Bird eye detection needs the optional extra: pip install 'outburst[subject]'"
        ) from exc

    from outburst.subject import preferred_device

    resolved = preferred_device() if device is None else device
    chosen = pose_device(resolved)
    if resolved == "mps":
        log.info("Bird eye detection is using the CPU so eye coordinates stay intact")
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
        Limitations: the model is not shared across threads. Arrays are passed through as BGR. Ultralytics treats a NumPy frame as BGR and swaps channels itself, so this method does not convert to RGB. Each frame is predicted at imgsz 640. A frame with no eye is searched again in overlapping 512 px windows, and the eye is mapped back onto the full preview. A short result list is padded with None. A prediction error on the full frames leaves those frames empty, and the window search still runs. A prediction error on the windows leaves those frames without an eye.
        """

        if not images:
            return []
        prepared = [np.ascontiguousarray(image) for image in images]
        results = self._predict(prepared)
        parsed = [
            eye_from_pose(result, image) if result is not None else None
            for result, image in zip(results, prepared, strict=False)
        ]
        if len(parsed) < len(prepared):
            parsed.extend([None] * (len(prepared) - len(parsed)))
        parsed = parsed[: len(prepared)]
        self._fill_from_tiles(prepared, parsed)
        return parsed

    def _predict(self, images: list[np.ndarray]) -> list[object | None]:
        """Run one pose prediction and align the results with the input list.

        Inputs: images, BGR arrays in batch order.
        Output: one result or None per image. A prediction error returns None for every image.
        Uses: find_many for the full preview and for the overlap windows.
        Limitations: the model is not shared across threads. This does not convert color. A short result list is padded with None.
        """

        predict = getattr(self._model, "predict")
        try:
            results = predict(images, verbose=False, device=self.device, imgsz=640)
        except Exception:
            log.warning("Bird eye detection failed for %d frame(s)", len(images), exc_info=True)
            return [None] * len(images)
        if not results:
            return [None] * len(images)
        aligned = list(results)
        if len(aligned) < len(images):
            aligned.extend([None] * (len(images) - len(aligned)))
        return aligned[: len(images)]

    def _fill_from_tiles(self, images: list[np.ndarray], parsed: list[Eye | None]) -> None:
        """Search overlapping windows for frames that still have no eye.

        Inputs: images, the BGR previews; parsed, the full-frame eyes, updated in place.
        Output: none. A window eye is written into parsed at that frame.
        Uses: find_many after the full-frame pass.
        Limitations: windows are skipped when every frame already has an eye, and when the preview is already 512 px or smaller on both sides. The window prediction runs on this same thread. A prediction error leaves the missing frames empty. Sharpness is measured on the full preview, not on the window.
        """

        tiles: list[np.ndarray] = []
        owners: list[tuple[int, int, int, np.ndarray]] = []
        for index, eye in enumerate(parsed):
            if eye is not None:
                continue
            for origin_x, origin_y, crop in _tiles(images[index]):
                tile = np.ascontiguousarray(crop)
                tiles.append(tile)
                owners.append((index, origin_x, origin_y, tile))
        if not tiles:
            return
        found = self._predict(tiles)
        best: dict[int, tuple[tuple[float, float], Eye]] = {}
        for (index, origin_x, origin_y, tile), result in zip(owners, found, strict=False):
            if result is None:
                continue
            eye = eye_from_pose(result, tile)
            if eye is None:
                continue
            moved = measure_eye(images[index], eye.x + origin_x, eye.y + origin_y, eye.radius)
            if moved is None:
                continue
            rank = (1.0 if moved.sharpness is not None else 0.0, moved.sharpness or 0.0)
            current = best.get(index)
            if current is None or rank > current[0]:
                best[index] = (rank, moved)
        for index, (_rank, eye) in best.items():
            parsed[index] = eye


def _tiles(image: np.ndarray) -> list[tuple[int, int, np.ndarray]]:
    """Cut a preview into overlapping squares when it is larger than one pose window.

    Inputs: image, a BGR preview.
    Output: origin x, origin y, and the window array. An empty list when both sides are already within TILE_EDGE.
    Uses: BirdEyeDetector._fill_from_tiles.
    Limitations: each window is at most 512 px on a side and steps by half that, so neighboring windows overlap. A preview that already fits in one window is not cut. The windows are views of the preview, not copies, and they are not resized.
    """

    height, width = image.shape[:2]
    if height <= TILE_EDGE and width <= TILE_EDGE:
        return []
    crops: list[tuple[int, int, np.ndarray]] = []
    for origin_y in _origins(height, TILE_EDGE):
        for origin_x in _origins(width, TILE_EDGE):
            crops.append((origin_x, origin_y, image[origin_y : origin_y + TILE_EDGE, origin_x : origin_x + TILE_EDGE]))
    return crops


def _origins(length: int, tile: int) -> list[int]:
    """Starting positions that cover one side of a preview with overlapping windows.

    Inputs: length, the preview side in pixels; tile, the window size.
    Output: a list of origins. A side that already fits returns one origin at 0.
    Uses: _tiles.
    Limitations: the step is half the window. The last origin is pulled back so the window stays inside the preview.
    """

    if length <= tile:
        return [0]
    step = max(1, tile // 2)
    origins = list(range(0, length - tile + 1, step))
    last = length - tile
    if origins[-1] != last:
        origins.append(last)
    return origins


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

    pad = 1
    height, width = gray.shape[:2]
    x0 = max(0, x - radius - pad)
    y0 = max(0, y - radius - pad)
    x1 = min(width, x + radius + pad + 1)
    y1 = min(height, y + radius + pad + 1)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    lap = cv2.Laplacian(crop, cv2.CV_64F)
    mask = np.zeros(crop.shape, dtype=np.uint8)
    cv2.circle(mask, (x - x0, y - y0), radius, 255, -1)
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
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()
    return np.asarray(value)
