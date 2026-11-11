"""Bird eye keypoints and sharpness. The pose model is not downloaded here.

``_portrait`` draws a synthetic eye. ``PortraitEyeFinder`` stands in for the pose model in pipeline tests. The other tests check disk size, blur, a low-confidence keypoint, a burst whose sharpest rated eye becomes the keeper, and that the detector keeps BGR order and can recover an eye from an overlapping window.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from outburst.eyes import (
    MIN_KEYPOINT_CONFIDENCE,
    MIN_RATED_DIAMETER,
    BirdEyeDetector,
    eye_from_pose,
    measure_eye,
    pose_device,
)
from outburst.subject import BIRD_EYE_OPTION, MODEL_OPTIONS, option_by_id
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


class PortraitEyeFinder:
    """Test double. A compact dark disk is an eye. A busy frame is not."""

    def find_many(self, images: list[np.ndarray]) -> list[object]:
        return [_portrait_eye(image) for image in images]


def _portrait_eye(image: np.ndarray) -> object | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = gray < 45
    fraction = float(dark.mean())
    if not (0.0008 <= fraction <= 0.08):
        return None
    ys, xs = np.nonzero(dark)
    radius = max(1, int(round(np.sqrt(len(xs) / np.pi))))
    return measure_eye(image, int(round(float(xs.mean()))), int(round(float(ys.mean()))), radius)


def _pose(xy: list[list[float]], conf: list[float], box: list[float] | None = None) -> object:
    class Result:
        keypoints = type("Keypoints", (), {"xy": np.asarray([xy]), "conf": np.asarray([conf])})()
        boxes = None

    result = Result()
    if box is not None:
        result.boxes = type(
            "Boxes",
            (),
            {"xyxy": np.asarray([box]), "conf": np.asarray([0.9]), "__len__": lambda self: 1},
        )()
    return result


def test_a_large_disk_is_rated_and_a_small_one_is_not() -> None:
    image = _portrait(36)
    large = measure_eye(image, 150, 100, (MIN_RATED_DIAMETER + 12) // 2)
    assert large is not None and large.rated and large.sharpness is not None and large.sharpness > 0
    small = measure_eye(image, 150, 100, (MIN_RATED_DIAMETER - 6) // 2)
    assert small is not None and small.rated is False and small.sharpness is None


def test_a_sharp_disk_outranks_a_blurred_copy() -> None:
    sharp = measure_eye(_portrait(36), 150, 100, 18)
    soft = measure_eye(_portrait(36, blur=21), 150, 100, 18)
    assert sharp is not None and sharp.sharpness is not None
    assert soft is not None and soft.sharpness is not None
    assert sharp.sharpness > soft.sharpness * 2


def test_pose_keeps_the_confident_eye_and_ignores_a_dim_keypoint() -> None:
    image = _portrait(36)
    blank = [0.0, 0.0]
    sharp = _pose(
        [blank, [150.0, 100.0], [40.0, 40.0], blank, blank, blank],
        [0.1, 0.9, 0.1, 0.1, 0.1, 0.1],
        [20.0, 20.0, 240.0, 200.0],
    )
    eye = eye_from_pose(sharp, image)
    assert eye is not None and eye.rated
    assert abs(eye.x - 150) <= 2 and abs(eye.y - 100) <= 2

    dim = _pose(
        [blank, [150.0, 100.0], blank, blank, blank, blank],
        [0.1, MIN_KEYPOINT_CONFIDENCE - 0.05, 0.1, 0.1, 0.1, 0.1],
        [20.0, 20.0, 240.0, 200.0],
    )
    assert eye_from_pose(dim, image) is None
    assert eye_from_pose(type("Empty", (), {"keypoints": None})(), image) is None


def test_pose_prefers_the_sharper_of_two_confident_eyes() -> None:
    image = _portrait(36)
    blank = [0.0, 0.0]
    result = _pose(
        [blank, [150.0, 100.0], [30.0, 30.0], blank, blank, blank],
        [0.2, 0.8, 0.95, 0.2, 0.2, 0.2],
        [10.0, 10.0, 250.0, 210.0],
    )
    eye = eye_from_pose(result, image)
    assert eye is not None
    assert abs(eye.x - 150) <= 2


def test_bird_eye_option_is_separate_from_the_subject_models() -> None:
    assert option_by_id("eye-pose-v0") == BIRD_EYE_OPTION
    assert BIRD_EYE_OPTION.label == "Bird eye detection"
    assert "huggingface.co/synthet/eye-pose-v0" in BIRD_EYE_OPTION.url
    assert all(item.id != BIRD_EYE_OPTION.id for item in MODEL_OPTIONS)


def test_bird_eye_finder_stays_on_the_calling_thread(tmp_path: Path) -> None:
    import threading

    folder = tmp_path / "batch"
    folder.mkdir()
    moment = datetime(2024, 7, 2, 9, 0, 0)
    for index in range(5):
        write_jpeg(folder / f"IMG_{index}.jpg", _portrait(36, seed=index + 2), moment)
    idents: set[int] = set()

    class RecordingFinder:
        def find_many(self, images: list[np.ndarray]) -> list[object]:
            idents.add(threading.get_ident())
            return [measure_eye(image, image.shape[1] // 2, image.shape[0] // 2, 12) for image in images]

    library = analyze_folder(
        folder,
        cache_dir=tmp_path / "cache",
        threshold=1.0,
        eye_scoring=True,
        eye_finder=RecordingFinder(),
    )
    assert len(library.analyses) == 5
    assert idents == {threading.get_ident()}
    assert all(item.metrics is not None and item.metrics.eye_sharpness is not None for item in library.analyses)


def test_bird_eye_without_weights_stays_off(tmp_path: Path) -> None:
    folder = tmp_path / "plain"
    folder.mkdir()
    write_jpeg(folder / "bird.jpg", sharp_frame(), datetime(2024, 7, 2, 9, 0, 0))
    library = analyze_folder(folder, cache_dir=tmp_path / "cache", threshold=1.0, eye_scoring=True)
    assert library.eye_active is False
    assert any("Bird eye detection stayed off" in note for note in library.warnings)


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

    library = analyze_folder(
        folder,
        cache_dir=tmp_path / "cache",
        threshold=1.0,
        eye_scoring=True,
        eye_finder=PortraitEyeFinder(),
    )
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


def test_pose_device_keeps_the_bird_eye_model_off_mps() -> None:
    assert pose_device("mps") == "cpu"
    assert pose_device("cpu") == "cpu"
    assert pose_device("cuda") == "cuda"


def test_a_window_eye_is_read_from_that_window() -> None:
    tall = np.zeros((600, 600, 3), dtype=np.uint8)
    tall[400, 20] = (255, 0, 0)
    short = np.zeros((120, 600, 3), dtype=np.uint8)

    class Model:
        def predict(self, images, verbose, device, imgsz):  # type: ignore[no-untyped-def]
            del verbose, device, imgsz
            results = []
            for frame in images:
                marked = (
                    frame.shape[0] == 512
                    and frame.shape[1] == 512
                    and int(frame[400, 20, 0]) == 255
                )
                if marked:
                    results.append(_pose_at(20, 400))
                else:
                    results.append(_pose([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], [0.0, 0.0, 0.0]))
            return results

    found = BirdEyeDetector(Model(), "cpu").find_many([tall, short])
    assert found[0] is not None
    assert found[0].x == 20
    assert found[0].y == 400
    assert found[1] is None


def test_finder_keeps_bgr_and_reads_an_eye_from_a_window() -> None:
    image = np.zeros((700, 1000, 3), dtype=np.uint8)
    image[500, 800] = (255, 0, 0)

    class Model:
        def __init__(self) -> None:
            self.seen: list[np.ndarray] = []

        def predict(self, images, verbose, device, imgsz):  # type: ignore[no-untyped-def]
            assert verbose is False
            assert imgsz == 640
            self.seen.extend(images)
            results = []
            for frame in images:
                height, width = frame.shape[:2]
                if height > 520 or width > 520 or not np.any(frame[:, :, 0] > 250):
                    results.append(_pose([], []))
                    continue
                ys, xs = np.nonzero(frame[:, :, 0] > 250)
                results.append(_pose_at(float(xs.mean()), float(ys.mean())))
            return results

    model = Model()
    found = BirdEyeDetector(model, "cpu").find_many([image])
    assert model.seen
    assert int(model.seen[0][500, 800, 0]) == 255
    assert int(model.seen[0][500, 800, 2]) == 0
    assert found[0] is not None
    assert abs(found[0].x - 800) <= 2
    assert abs(found[0].y - 500) <= 2


def _pose_at(x: float, y: float) -> object:
    blank = [0.0, 0.0]
    return _pose(
        [blank, blank, [x, y], blank, blank, blank],
        [0.0, 0.0, 0.95, 0.0, 0.0, 0.0],
        [x - 80, y - 80, x + 80, y + 80],
    )
