"""Subject detection stays optional, and weights download only through ensure_model.

The tests check the device name, the Ultralytics install command, a missing or local weights file, the models folder, download, a cancelled or failed download, and ``detect_many`` on one batch. ``_weights_bytes``, ``_Body``, and ``_opener`` fake the download response. A fake pip process covers the install failure and cancel paths.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from outburst.subject import (
    MIN_MODEL_BYTES,
    ModelCancelled,
    SubjectRuntimeCancelled,
    ensure_model,
    install_subject_runtime,
    load_yolo_detector,
    model_is_ready,
    models_dir,
    option_by_id,
    preferred_device,
    subject_runtime_can_install,
    subject_runtime_install_command,
    subject_runtime_ready,
)


def test_preferred_device_is_a_known_name() -> None:
    assert preferred_device() in {"cpu", "mps", "cuda"}


def test_subject_runtime_install_command_targets_this_interpreter() -> None:
    import sys

    assert subject_runtime_can_install() is True
    command = subject_runtime_install_command()
    assert command[0] == sys.executable
    assert command[1:3] == ["-m", "pip"]
    assert command[-1] == "ultralytics>=8.3"


def test_subject_runtime_ready_matches_the_import() -> None:
    try:
        import ultralytics  # noqa: F401

        expected = True
    except ImportError:
        expected = False
    assert subject_runtime_ready() is expected


def test_subject_hardware_support_matches_the_published_builds() -> None:
    from outburst.subject import SUBJECT_INSTALL_FREE_BYTES, subject_hardware_support

    plenty = SUBJECT_INSTALL_FREE_BYTES
    short = SUBJECT_INSTALL_FREE_BYTES - 1
    ok = subject_hardware_support(
        system="linux",
        machine="x86_64",
        bits=64,
        free_bytes=plenty,
    )
    assert ok.compatible is True
    apple = subject_hardware_support(
        system="darwin",
        machine="arm64",
        bits=64,
        macos_version="15.2",
        free_bytes=plenty,
    )
    assert apple.compatible is True
    windows = subject_hardware_support(
        system="win32",
        machine="AMD64",
        bits=64,
        windows_major=10,
        free_bytes=plenty,
    )
    assert windows.compatible is True
    intel_mac = subject_hardware_support(
        system="darwin",
        machine="x86_64",
        bits=64,
        macos_version="15.2",
        free_bytes=plenty,
    )
    assert intel_mac.compatible is False
    assert "Intel" in intel_mac.explanation
    assert "will not install" in intel_mac.explanation
    assert "download the models" in intel_mac.explanation
    old_mac = subject_hardware_support(
        system="darwin",
        machine="arm64",
        bits=64,
        macos_version="13.6",
        free_bytes=plenty,
    )
    assert old_mac.compatible is False
    assert "macOS 14" in old_mac.explanation
    arm_pc = subject_hardware_support(
        system="win32",
        machine="ARM64",
        bits=64,
        windows_major=10,
        free_bytes=plenty,
    )
    assert arm_pc.compatible is False
    assert "64-bit Intel and AMD" in arm_pc.explanation
    narrow = subject_hardware_support(system="linux", machine="x86_64", bits=32, free_bytes=plenty)
    assert narrow.compatible is False
    assert "32-bit" in narrow.explanation
    tight = subject_hardware_support(
        system="linux",
        machine="aarch64",
        bits=64,
        free_bytes=short,
    )
    assert tight.compatible is False
    assert "5 GB" in tight.explanation
    installed = subject_hardware_support(
        system="linux",
        machine="aarch64",
        bits=64,
        free_bytes=short,
        needs_install=False,
    )
    assert installed.compatible is True


def test_internet_connection_reports_a_reachable_host(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error
    from email.message import Message

    from outburst.subject import internet_connection

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def read(self, _size: int) -> bytes:
            return b""

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    assert internet_connection().online is True

    def http_error(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError("https://pypi.org", 405, "Method Not Allowed", Message(), None)

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", http_error)
    assert internet_connection().online is True


def test_internet_connection_reports_an_offline_computer(monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    from outburst.subject import OFFLINE_EXPLANATION, internet_connection

    def offline(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", offline)
    status = internet_connection()
    assert status.online is False
    assert status.explanation == OFFLINE_EXPLANATION
    assert "not downloaded" in status.explanation


def test_subject_license_text_names_the_agpl_terms() -> None:
    from outburst.subject import SUBJECT_LICENSE_TEXT

    assert "AGPL-3.0" in SUBJECT_LICENSE_TEXT
    assert "eye_pose_v0.pt" in SUBJECT_LICENSE_TEXT
    assert "does not install" in SUBJECT_LICENSE_TEXT
    assert "does not download" in SUBJECT_LICENSE_TEXT


def test_frozen_app_does_not_pip_install_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert subject_runtime_can_install() is False
    with pytest.raises(RuntimeError, match="not included"):
        install_subject_runtime()


def test_install_subject_runtime_reports_a_pip_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class Proc:
        returncode = 1

        def poll(self) -> int:
            return self.returncode

        def communicate(self) -> tuple[str, str]:
            return "", "Could not find a version"

    monkeypatch.setattr("subprocess.Popen", lambda *_args, **_kwargs: Proc())
    with pytest.raises(RuntimeError, match="Could not find a version"):
        install_subject_runtime()


def test_install_subject_runtime_stops_when_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    class Proc:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            assert self.returncode is not None
            return self.returncode

        def communicate(self) -> tuple[str, str]:
            raise AssertionError("cancelled install must not read pip output")

    monkeypatch.setattr("subprocess.Popen", lambda *_args, **_kwargs: Proc())
    with pytest.raises(SubjectRuntimeCancelled):
        install_subject_runtime(should_cancel=lambda: True)


def test_missing_model_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yolo_detector(tmp_path / "missing.pt")


def test_local_file_does_not_download_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("load_yolo_detector must not download")

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", fail)
    model = tmp_path / "local.pt"
    model.write_bytes(b"not a real model")
    with pytest.raises(Exception) as caught:
        load_yolo_detector(model)
    assert not isinstance(caught.value, AssertionError)


def test_models_dir_uses_the_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "chosen"
    monkeypatch.setenv("OUTBURST_MODELS", str(target))
    assert models_dir() == target.resolve()
    assert target.is_dir()


def test_models_dir_finds_the_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTBURST_MODELS", raising=False)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "outburst"\n', encoding="utf-8")
    monkeypatch.chdir(project)
    assert models_dir() == project / "models"
    assert (project / "models").is_dir()


def test_models_dir_falls_back_to_the_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OUTBURST_MODELS", raising=False)
    monkeypatch.setattr("outburst.subject._discover_project_models", lambda: None)
    monkeypatch.setattr("outburst.subject.default_cache_dir", lambda: tmp_path / "cache")
    assert models_dir() == tmp_path / "cache" / "models"
    assert (tmp_path / "cache" / "models").is_dir()


def test_ready_file_is_not_downloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    option = option_by_id("yolo11n")
    assert option is not None
    ready = tmp_path / option.filename
    ready.write_bytes(_weights_bytes())

    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a ready model must not be downloaded again")

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", fail)
    assert ensure_model(option, tmp_path) == ready
    assert model_is_ready(ready)


def test_missing_model_is_downloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    option = option_by_id("yolo11s")
    assert option is not None
    body = _weights_bytes()
    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", _opener(body, option.url))
    seen: list[tuple[int, int | None]] = []
    path = ensure_model(option, tmp_path, progress=lambda done, total: seen.append((done, total)))
    assert path == tmp_path / "yolo11s.pt"
    assert path.read_bytes() == body
    assert model_is_ready(path)
    assert seen[-1] == (len(body), len(body))
    assert not (tmp_path / "yolo11s.pt.part").exists()


def test_small_file_is_replaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    option = option_by_id("yolov8n")
    assert option is not None
    (tmp_path / option.filename).write_bytes(b"PK")
    body = _weights_bytes()
    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", _opener(body, option.url))
    path = ensure_model(option, tmp_path)
    assert path.read_bytes() == body


def test_unusable_download_is_removed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    option = option_by_id("yolov8s")
    assert option is not None
    monkeypatch.setattr(
        "outburst.subject.urllib.request.urlopen",
        _opener(b"NO" * MIN_MODEL_BYTES, option.url),
    )
    with pytest.raises(RuntimeError, match="not a usable model"):
        ensure_model(option, tmp_path)
    assert not (tmp_path / option.filename).exists()
    assert not (tmp_path / f"{option.filename}.part").exists()


def test_cancel_removes_the_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    option = option_by_id("yolo11n")
    assert option is not None
    reads = {"count": 0}

    def cancel() -> bool:
        return reads["count"] >= 1

    def opener(*_args: object, **_kwargs: object) -> _Body:
        return _Body(_weights_bytes(), reads)

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", opener)
    with pytest.raises(ModelCancelled):
        ensure_model(option, tmp_path, should_cancel=cancel)
    assert not (tmp_path / option.filename).exists()
    assert not (tmp_path / f"{option.filename}.part").exists()


def test_download_error_removes_the_partial_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import urllib.error

    option = option_by_id("yolo11n")
    assert option is not None

    def fail(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", fail)
    with pytest.raises(RuntimeError, match="offline"):
        ensure_model(option, tmp_path)
    assert not (tmp_path / option.filename).exists()
    assert not (tmp_path / f"{option.filename}.part").exists()


def test_detect_many_reads_a_batch_on_one_call() -> None:
    import numpy as np

    from outburst.subject import YoloSubjectDetector

    class Boxes:
        xyxy = [[1, 2, 11, 14], [0, 0, 3, 3]]
        conf = [0.95, 0.4]
        cls = [14, 14]

        def __len__(self) -> int:
            return len(self.cls)

    class Result:
        boxes = Boxes()
        names = {14: "bird"}

    class Model:
        def __init__(self) -> None:
            self.calls = 0

        def predict(self, images, verbose, device, imgsz):  # type: ignore[no-untyped-def]
            self.calls += 1
            self.batch = len(images)
            assert verbose is False
            assert device == "cpu"
            assert imgsz == 640
            return [Result() for _image in images]

    model = Model()
    detector = YoloSubjectDetector(model, "cpu")
    blank = np.zeros((12, 12, 3), dtype=np.uint8)
    found = detector.detect_many([blank, blank])
    assert model.calls == 1
    assert model.batch == 2
    assert found[0] is not None and found[0].label == "bird"
    assert found[0].box == (1, 2, 10, 12)
    assert found[0].confidence == 0.95
    assert found[1] is not None and found[1].label == "bird"
    assert found[1].box == (1, 2, 10, 12)
    single = detector.detect(blank)
    assert single is not None and single.label == "bird"
    assert model.calls == 2


def test_unknown_model_id() -> None:
    assert option_by_id("missing") is None
    assert option_by_id("yolo11n") is not None


def _weights_bytes() -> bytes:
    return b"PK" + b"\x00" * MIN_MODEL_BYTES


class _Body:
    def __init__(self, payload: bytes, reads: dict[str, int] | None = None) -> None:
        self._payload = payload
        self._offset = 0
        self._reads = reads
        self.headers = {"Content-Length": str(len(payload))}

    def read(self, size: int) -> bytes:
        if self._reads is not None:
            self._reads["count"] += 1
        chunk = self._payload[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        return None


def _opener(payload: bytes, expected_url: str):
    def opener(request: object, timeout: int = 60) -> _Body:
        assert timeout == 60
        assert getattr(request, "full_url", None) == expected_url
        return _Body(payload)

    return opener
