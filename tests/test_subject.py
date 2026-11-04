"""Subject detection stays optional, and weights download only through ensure_model."""

from __future__ import annotations

from pathlib import Path

import pytest

from outburst.subject import (
    MIN_MODEL_BYTES,
    ModelCancelled,
    ensure_model,
    load_yolo_detector,
    model_is_ready,
    models_dir,
    option_by_id,
    preferred_device,
)


def test_preferred_device_is_a_known_name() -> None:
    assert preferred_device() in {"cpu", "mps", "cuda"}


def test_missing_model_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yolo_detector(tmp_path / "missing.pt")


def test_local_file_does_not_download_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("load_yolo_detector must not download")

    monkeypatch.setattr("outburst.subject.urllib.request.urlopen", fail)
    model = tmp_path / "local.pt"
    model.write_bytes(b"not a real model")
    with pytest.raises(ImportError):
        load_yolo_detector(model)


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
