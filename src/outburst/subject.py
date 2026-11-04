"""Optional subject detection.

The core app scores the whole preview. When a YOLO model is available,
Outburst can measure sharpness inside the animal's bounding box instead.
Weights live in the models folder. A missing file is downloaded there when
subject scoring needs it. Loading a detector never fetches weights by itself.
If detection is unavailable or finds no animal, scoring falls back to the
full frame.
"""

from __future__ import annotations

import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from outburst.cache import default_cache_dir

log = logging.getLogger(__name__)

# COCO classes that cover typical wildlife and nature subjects.
ANIMAL_CLASS_IDS = frozenset({14, 15, 16, 17, 18, 19, 20, 21, 22, 23})

_ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
MIN_MODEL_BYTES = 1_000_000
DEFAULT_MODEL_ID = "yolo11n"


@dataclass(frozen=True)
class ModelOption:
    """One downloadable subject model."""

    id: str
    label: str
    filename: str
    url: str
    note: str


MODEL_OPTIONS: tuple[ModelOption, ...] = (
    ModelOption(
        "yolo11n",
        "YOLO11 Nano",
        "yolo11n.pt",
        f"{_ASSET_BASE}/yolo11n.pt",
        "Fast, about 6 MB",
    ),
    ModelOption(
        "yolo11s",
        "YOLO11 Small",
        "yolo11s.pt",
        f"{_ASSET_BASE}/yolo11s.pt",
        "More accurate, about 19 MB",
    ),
    ModelOption(
        "yolov8n",
        "YOLOv8 Nano",
        "yolov8n.pt",
        f"{_ASSET_BASE}/yolov8n.pt",
        "Fast, about 6 MB",
    ),
    ModelOption(
        "yolov8s",
        "YOLOv8 Small",
        "yolov8s.pt",
        f"{_ASSET_BASE}/yolov8s.pt",
        "More accurate, about 22 MB",
    ),
)


class ModelCancelled(Exception):
    """The photographer cancelled a model download."""


def option_by_id(model_id: str) -> ModelOption | None:
    for option in MODEL_OPTIONS:
        if option.id == model_id:
            return option
    return None


def models_dir() -> Path:
    """Folder where subject weights are stored. Created on first use."""

    override = os.environ.get("OUTBURST_MODELS")
    if override:
        path = Path(override).expanduser().resolve()
    else:
        discovered = _discover_project_models()
        path = discovered if discovered is not None else default_cache_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(option: ModelOption, directory: Path | None = None) -> Path:
    root = directory if directory is not None else models_dir()
    return root / option.filename


def model_is_ready(path: Path) -> bool:
    """True when path looks like a downloaded PyTorch weights file."""

    try:
        if not path.is_file() or path.stat().st_size < MIN_MODEL_BYTES:
            return False
        with path.open("rb") as handle:
            return handle.read(2) == b"PK"
    except OSError:
        return False


def ensure_model(
    option: ModelOption,
    directory: Path | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> Path:
    """Return the weights file, downloading it when the models folder does not have it."""

    root = directory if directory is not None else models_dir()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / option.filename
    if model_is_ready(destination):
        return destination
    partial = destination.with_name(destination.name + ".part")
    try:
        log.info("Downloading %s into %s", option.filename, root)
        _stream_download(option.url, partial, progress, should_cancel)
        if should_cancel is not None and should_cancel():
            raise ModelCancelled()
        partial.replace(destination)
    except ModelCancelled:
        partial.unlink(missing_ok=True)
        raise
    except Exception as exc:
        partial.unlink(missing_ok=True)
        if not model_is_ready(destination):
            destination.unlink(missing_ok=True)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Could not download {option.filename}: {exc}") from exc
    if not model_is_ready(destination):
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"Downloaded file is not a usable model: {option.filename}")
    return destination


def _discover_project_models() -> Path | None:
    starts = [Path.cwd(), Path(__file__).resolve().parents[1].parent]
    seen: set[Path] = set()
    for start in starts:
        try:
            current = start.resolve()
        except OSError:
            continue
        while current not in seen:
            seen.add(current)
            marker = current / "pyproject.toml"
            if marker.is_file() and _is_outburst_project(marker):
                return current / "models"
            if current.parent == current:
                break
            current = current.parent
    return None


def _is_outburst_project(marker: Path) -> bool:
    try:
        text = marker.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(line.strip() in {'name = "outburst"', "name = 'outburst'"} for line in text.splitlines())


def _stream_download(
    url: str,
    destination: Path,
    progress: Callable[[int, int | None], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Outburst"})
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Could not download the model ({exc.code})") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download the model: {exc.reason}") from exc
    try:
        total = _content_length(response)
        downloaded = 0
        with destination.open("wb") as handle:
            while True:
                if should_cancel is not None and should_cancel():
                    raise ModelCancelled()
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if progress is not None:
                    progress(downloaded, total)
    finally:
        close = getattr(response, "close", None)
        if callable(close):
            close()


def _content_length(response: object) -> int | None:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    raw = getter("Content-Length")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


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
    """Load a YOLO weights file that is already on disk. This does not download anything."""

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
