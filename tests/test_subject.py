"""Subject detection stays optional and offline."""

from __future__ import annotations

from pathlib import Path

import pytest

from outburst.subject import load_yolo_detector, preferred_device


def test_preferred_device_is_a_known_name() -> None:
    assert preferred_device() in {"cpu", "mps", "cuda"}


def test_missing_model_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_yolo_detector(tmp_path / "missing.pt")


def test_local_file_does_not_download_weights(tmp_path: Path) -> None:
    model = tmp_path / "local.pt"
    model.write_bytes(b"not a real model")
    with pytest.raises(ImportError):
        load_yolo_detector(model)
