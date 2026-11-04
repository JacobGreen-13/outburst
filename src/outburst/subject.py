"""Optional subject detection.

The core app scores the whole preview. When a local YOLO model is available,
Outburst can measure sharpness inside the animal's bounding box instead.
Nothing is downloaded. If detection is unavailable or finds no animal, scoring
falls back to the full frame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

# COCO classes that cover typical wildlife and nature subjects.
ANIMAL_CLASS_IDS = frozenset({14, 15, 16, 17, 18, 19, 20, 21, 22, 23})


class Detection:
    """One subject box in preview-pixel coordinates: x, y, width, height."""

    def __init__(self, box: tuple[int, int, int, int], label: str, confidence: float) -> None:
        self.box = box
        self.label = label
        self.confidence = confidence


class SubjectDetector(Protocol):
    def detect(self, image: np.ndarray) -> Detection | None:
        """Return the best animal box, or None when the frame has no subject."""


def preferred_device() -> str:
    """Pick MPS on Apple Silicon, CUDA on NVIDIA, otherwise CPU."""

    try:
        import torch
    except Exception:
        return "cpu"
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and cuda.is_available():
            return "cuda"
    except Exception:
        log.debug("Could not query the torch device", exc_info=True)
    return "cpu"


def make_subject_key(model_path: Path) -> str:
    """Cache key that changes when the model file changes."""

    stat = model_path.stat()
    return f"{model_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"


def load_yolo_detector(model_path: Path, device: str | None = None) -> SubjectDetector:
    """Load a local YOLO weights file. The file must already be on disk."""

    if not model_path.is_file():
        raise FileNotFoundError(f"Subject model not found: {model_path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Subject detection needs the optional extra: pip install 'outburst[subject]'"
        ) from exc
    chosen = device or preferred_device()
    log.info("Loading subject model %s on %s", model_path, chosen)
    return YoloSubjectDetector(YOLO(str(model_path)), chosen)


class YoloSubjectDetector:
    """Ultralytics YOLO wrapper limited to animal and bird classes."""

    def __init__(self, model: object, device: str) -> None:
        self._model = model
        self.device = device

    def detect(self, image: np.ndarray) -> Detection | None:
        import cv2

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predict = getattr(self._model, "predict")
        results = predict(rgb, verbose=False, device=self.device, imgsz=640)
        if not results:
            return None
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        xyxy = _as_numpy(boxes.xyxy)
        confidences = _as_numpy(boxes.conf)
        classes = _as_numpy(boxes.cls).astype(int)
        names = getattr(result, "names", {}) or {}
        best: tuple[float, Detection] | None = None
        for index, class_id in enumerate(classes):
            if int(class_id) not in ANIMAL_CLASS_IDS:
                continue
            confidence = float(confidences[index])
            if best is not None and confidence <= best[0]:
                continue
            x1, y1, x2, y2 = (float(value) for value in xyxy[index][:4])
            box = (int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1)))
            label = str(names.get(int(class_id), class_id))
            best = (confidence, Detection(box, label, confidence))
        if best is None:
            return None
        return best[1]


def _as_numpy(value: object) -> np.ndarray:
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()
    return np.asarray(value)
