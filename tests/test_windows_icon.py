"""The Windows icon writer stores PNG images in an ICO file.

``write_ico`` is the packager used before PyInstaller runs. The test checks the icon directory and the stored PNG bytes without opening Qt.
"""

from __future__ import annotations

import importlib.util
import struct
from pathlib import Path


def _load_make_icon():
    path = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "make_icon.py"
    spec = importlib.util.spec_from_file_location("outburst_make_icon", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_write_ico_stores_each_png(tmp_path: Path) -> None:
    make_icon = _load_make_icon()
    destination = tmp_path / "outburst.ico"
    images = [(16, b"\x89PNG\r\n\x1a\nsmall"), (256, b"\x89PNG\r\n\x1a\nlarge")]
    make_icon.write_ico(destination, images)
    data = destination.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind, count) == (0, 1, 2)
    first = struct.unpack_from("<BBBBHHII", data, 6)
    second = struct.unpack_from("<BBBBHHII", data, 22)
    assert first[0] == 16
    assert second[0] == 0
    assert data[first[7] : first[7] + first[6]] == images[0][1]
    assert data[second[7] : second[7] + second[6]] == images[1][1]
