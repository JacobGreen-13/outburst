"""Optional subject detection.

The core app scores the whole preview. When a YOLO model is available,
Outburst can measure sharpness inside the animal's bounding box instead.
Weights live in the models folder. A missing file is downloaded there when
subject scoring needs it. Loading a detector never fetches weights by itself.
If detection is unavailable or finds no animal, scoring falls back to the
full frame.

``option_by_id``, ``models_dir``, ``model_path``, and ``model_is_ready`` find a weights file. ``ensure_model`` downloads a missing file. ``subject_runtime_ready`` reports whether Ultralytics imports. ``subject_runtime_can_install`` reports whether this process can pip-install that library. ``install_subject_runtime`` installs that library into the running Python. ``load_yolo_detector`` only opens a file that is already there. ``YoloSubjectDetector.detect`` and ``detect_many`` return one animal box per image. ``preferred_device`` tries MPS, then CUDA, then CPU. ``make_subject_key`` is the cache suffix for the chosen model. ``BIRD_EYE_OPTION`` is the separate open-weight bird eye pose file. It is not one of the subject dropdown models.
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


# Open weights: synthet/eye-pose-v0, YOLO11n-pose, AGPL-3.0. About 5.6 MB.
# Trained on CUB-200-2011 bird photos. Not a mammal eye model.
BIRD_EYE_OPTION = ModelOption(
    "eye-pose-v0",
    "Bird eye detection",
    "eye_pose_v0.pt",
    "https://huggingface.co/synthet/eye-pose-v0/resolve/main/eye_pose_v0.pt",
    "Bird eyes, about 6 MB",
)

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
    """Look up one catalog entry by its model id.

    Inputs: model_id, a catalog id such as yolo11n, or eye-pose-v0.
    Output: the matching ModelOption, or None when the id is not known.
    Uses: the subject-model dropdown, and the bird-eye download.
    Limitations: the four subject ids and the bird-eye id match. This does not download weights or load a detector. The bird-eye option is not part of MODEL_OPTIONS.
    """

    if model_id == BIRD_EYE_OPTION.id:
        return BIRD_EYE_OPTION
    for option in MODEL_OPTIONS:
        if option.id == model_id:
            return option
    return None


def models_dir() -> Path:
    """Folder where subject weights are stored. Created on first use.

    Inputs: none. OUTBURST_MODELS overrides the location when that environment variable is set.
    Output: the models directory, created if it was missing.
    Uses: model downloads and the path passed to the detector loader.
    Limitations: creating the directory is the side effect. Photographs are not written. Without the override, a project models folder is used when the checkout is found, and the cache models folder otherwise.
    """

    override = os.environ.get("OUTBURST_MODELS")
    if override:
        path = Path(override).expanduser().resolve()
    else:
        discovered = _discover_project_models()
        path = discovered if discovered is not None else default_cache_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_path(option: ModelOption, directory: Path | None = None) -> Path:
    """Path of one catalog weights file inside the models folder.

    Inputs: option, a catalog entry; directory, an optional folder. The models folder is used when directory is omitted.
    Output: the path of option.filename in that folder. The file is not created.
    Uses: the download check and the loader.
    Limitations: this does not download or open the weights. A custom directory is not created.
    """

    root = directory if directory is not None else models_dir()
    return root / option.filename


def model_is_ready(path: Path) -> bool:
    """True when path looks like a downloaded PyTorch weights file.

    Inputs: path, a candidate weights file.
    Output: True when the file exists, is at least MIN_MODEL_BYTES, and starts with the PK zip signature.
    Uses: ensure_model before and after a download.
    Limitations: the check is size plus the first two bytes. It does not load the network. The Ultralytics license is separate from this check.
    """

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
    """Return the weights file, downloading it when the models folder does not have it.

    Inputs: option, the catalog entry; directory, an optional models folder; progress, an optional callback of bytes downloaded and total size; should_cancel, an optional callback that is true when the photographer cancels.
    Output: the path of the ready weights file.
    Uses: a subject-scoring rescan, which downloads a missing model before the scan loads it.
    Limitations: a ready file is returned without a download. Cancel deletes the partial file and raises ModelCancelled. Subject weights are the official Ultralytics YOLO files. The bird-eye file is eye-pose-v0. Both are AGPL-3.0, separate from Outburst. Photographs are not uploaded.
    """

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
    """Find a models folder beside an Outburst checkout.

    Inputs: none. The walk starts at the working directory and at this file's project root.
    Output: the project models path when a pyproject.toml names outburst, otherwise None. The folder is not created here.
    Uses: models_dir when OUTBURST_MODELS is unset.
    Limitations: the first matching project wins. A checkout whose name is not outburst is ignored.
    """

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
    """Tell whether a pyproject.toml belongs to this project.

    Inputs: marker, the path of a pyproject.toml.
    Output: True when a line sets the project name to outburst.
    Uses: _discover_project_models.
    Limitations: the match is a stripped name line, with double or single quotes. A file that cannot be read returns False.
    """

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
    """Download a weights file to a partial path, reporting progress and stopping when cancelled.

    Inputs: url, the catalog URL; destination, the partial file to write; progress, an optional byte callback; should_cancel, an optional cancel check.
    Output: none. The partial file is written. ModelCancelled is raised when the photographer cancels.
    Uses: ensure_model.
    Limitations: the request times out after 60 seconds and reads 256 KB at a time. HTTP and URL errors become RuntimeError. The caller renames the partial file only after this returns.
    """

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
    """Read a positive Content-Length from a download response.

    Inputs: response, an object that may have headers.
    Output: the length in bytes, or None when the header is missing or not a positive integer.
    Uses: _stream_download so the progress callback can show a total.
    Limitations: a missing header leaves the total unknown. The body is not read.
    """

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
        """Store one animal box in preview pixels.

        Inputs: box, x, y, width, and height; label, the class name; confidence, the detector score.
        Output: none. The three fields are stored on the instance.
        Uses: YoloSubjectDetector when it returns the best animal in a preview.
        Limitations: the box is not clipped here. Coordinates are preview pixels on the 1024 px analysis image, not the original RAW.
        """

        self.box = box
        self.label = label
        self.confidence = confidence


class SubjectDetector(Protocol):
    def detect(self, image: np.ndarray) -> Detection | None:
        """Return the best animal box, or None when the frame has no subject.

        Inputs: image, one BGR preview.
        Output: a Detection, or None.
        Uses: the scan pipeline when a detector has no detect_many method, and YoloSubjectDetector.detect.
        Limitations: a detector is not shared across threads. The caller runs detection on one thread. No animal means the frame keeps full-frame sharpness.
        """


def preferred_device() -> str:
    """Pick MPS on Apple Silicon, CUDA on NVIDIA, otherwise CPU.

    Inputs: none.
    Output: `mps`, `cuda`, or `cpu`.
    Uses: load_yolo_detector when the caller does not pass a device.
    Limitations: a missing torch install, or a failed device query, returns cpu. The choice belongs to one detector and is not shared across threads.
    """

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
    """Cache key that changes when the model file changes.

    Inputs: model_path, the weights file already on disk.
    Output: a string of the resolved path, the mtime in nanoseconds, and the size.
    Uses: the scan cache so a replaced weights file is not scored from old rows.
    Limitations: the file must exist so stat can run. This does not load the model. Bird eye detection appends the same kind of suffix for its own weights file.
    """

    stat = model_path.stat()
    return f"{model_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}"


SUBJECT_PIP_SPEC = "ultralytics>=8.3"


class SubjectRuntimeCancelled(Exception):
    """The Ultralytics install was cancelled before pip finished."""


def subject_runtime_ready() -> bool:
    """
    Report whether Ultralytics can be imported in this process.

    Inputs: none.
    Output: True when ``import ultralytics`` succeeds, otherwise False.
    Uses: deciding whether subject detection and bird eye detection can load.
    Limitations: a failed import is treated as missing. This does not install anything and does not check that a weights file is present.
    """

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def subject_runtime_can_install() -> bool:
    """Report whether this process can pip-install Ultralytics into itself.

    Inputs: none.
    Output: False for a frozen installer build. True for a normal Python process.
    Uses: the review window before it offers to install subject detection.
    Limitations: this does not check that pip or a network connection is available. A frozen build that already imported Ultralytics still returns False here; callers also check ``subject_runtime_ready``.
    """

    import sys

    return not bool(getattr(sys, "frozen", False))


def subject_runtime_install_command() -> list[str]:
    """
    Build the pip command that installs Ultralytics into this interpreter.

    Inputs: none.
    Output: the argument list, starting with this process's Python and ending with the ultralytics requirement.
    Uses: the review window and tests that check the install command.
    Limitations: the command installs the library only. It does not download YOLO weights.
    """

    import sys

    return [sys.executable, "-m", "pip", "install", SUBJECT_PIP_SPEC]


def install_subject_runtime(should_cancel: Callable[[], bool] | None = None) -> None:
    """
    Install Ultralytics into the Python that is running Outburst.

    Inputs: should_cancel, an optional check that is polled while pip runs. None never cancels.
    Output: no return value. Raises SubjectRuntimeCancelled when the check asks to stop, and RuntimeError when pip exits with an error.
    Uses: the review window the first time subject or bird eye detection is turned on without Ultralytics.
    Limitations: a frozen installer build raises RuntimeError and does not start pip. A normal install downloads PyTorch and can take several minutes. It does not download model weights. The caller should stay off the window thread.
    """

    import subprocess

    if not subject_runtime_can_install():
        raise RuntimeError(
            "This install scores the whole frame. "
            "Subject-aware sharpness and bird eye detection are not included."
        )
    process = subprocess.Popen(
        subject_runtime_install_command(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while process.poll() is None:
        if should_cancel is not None and should_cancel():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise SubjectRuntimeCancelled()
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            continue
        break
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        detail = (stderr or stdout or "pip install failed").strip()
        raise RuntimeError(detail[-800:])


def load_yolo_detector(model_path: Path, device: str | None = None) -> SubjectDetector:
    """Load a YOLO weights file that is already on disk. This does not download anything.

    Inputs: model_path, the .pt file; device, an optional torch device. preferred_device is used when device is omitted.
    Output: a YoloSubjectDetector bound to that file and device.
    Uses: analyze_folder when subject scoring is on and no detector was passed in.
    Limitations: a missing file raises FileNotFoundError, and a missing ultralytics install raises ImportError. The weights are Ultralytics YOLO files under the AGPL-3.0 license, separate from Outburst. The returned detector must stay on one thread. Device order is MPS, then CUDA, then CPU.
    """

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
        """Keep one loaded YOLO model and the device it runs on.

        Inputs: model, an ultralytics YOLO instance; device, `mps`, `cuda`, or `cpu`.
        Output: none. The model and device are stored on the instance.
        Uses: load_yolo_detector.
        Limitations: the instance is not shared across threads. Loading it does not download weights. The Ultralytics AGPL-3.0 license applies to the model.
        """

        self._model = model
        self.device = device

    def detect(self, image: np.ndarray) -> Detection | None:
        """Detect the animal in one preview by running a one-image batch.

        Inputs: image, a BGR preview.
        Output: the Detection for that image, or None when the frame has no animal.
        Uses: callers that ask for a single frame, including the pipeline fallback.
        Limitations: the call goes through detect_many on this same thread. The model is not shared across threads.
        """

        found = self.detect_many([image])
        if not found:
            return None
        return found[0]

    def detect_many(self, images: list[np.ndarray]) -> list[Detection | None]:
        """Score a small batch on this one model. Callers keep this on one thread.

        Inputs: images, BGR previews in batch order.
        Output: one Detection or None per image, in the same order. An empty input returns an empty list.
        Uses: the scan pipeline, which calls this on the coordinating thread.
        Limitations: the model is not shared across threads. Arrays are passed through as BGR and predicted at imgsz 640 on this detector's device. Ultralytics treats a NumPy frame as BGR and swaps channels itself. Only COCO animal and bird classes are kept. A short result list is padded with None.
        """

        if not images:
            return []
        prepared = [np.ascontiguousarray(image) for image in images]
        predict = getattr(self._model, "predict")
        results = predict(prepared, verbose=False, device=self.device, imgsz=640)
        if not results:
            return [None] * len(images)
        parsed = [_best_animal(result) for result in results]
        if len(parsed) < len(images):
            parsed.extend([None] * (len(images) - len(parsed)))
        return parsed[: len(images)]


def _best_animal(result: object) -> Detection | None:
    """Pick the highest-confidence animal or bird box from one YOLO result.

    Inputs: result, one Ultralytics prediction.
    Output: a Detection in preview pixels, or None when no animal class is present.
    Uses: detect_many.
    Limitations: classes outside the COCO animal and bird set are ignored. The box is the highest confidence, not the largest area.
    """

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
    """Copy a detector tensor onto a NumPy array.

    Inputs: value, a tensor or an array-like box field.
    Output: a NumPy array. A cpu method is called first when it exists.
    Uses: _best_animal so MPS or CUDA boxes can be read as pixels.
    Limitations: this moves values to the CPU for reading. It does not run the model.
    """

    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        value = numpy_fn()
    return np.asarray(value)
