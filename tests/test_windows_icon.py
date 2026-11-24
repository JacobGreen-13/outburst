"""The Windows icon writer stores 32-bit bitmaps in an ICO file.

``dib_32`` packs one RGBA image. ``write_ico`` is the packager used before PyInstaller runs. The desktop shortcut points at that file. The tests check the bitmap layout and the installer script without opening Qt, then one test scales the real artwork.
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


def test_write_ico_stores_each_bitmap(tmp_path: Path) -> None:
    make_icon = _load_make_icon()
    destination = tmp_path / "outburst.ico"
    images = [(16, b"small-bitmap"), (256, b"large-bitmap")]
    make_icon.write_ico(destination, images)
    data = destination.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert (reserved, kind, count) == (0, 1, 2)
    first = struct.unpack_from("<BBBBHHII", data, 6)
    second = struct.unpack_from("<BBBBHHII", data, 22)
    assert first[0] == 16 and first[4] == 1 and first[5] == 32
    assert second[0] == 0 and second[5] == 32
    assert data[first[7] : first[7] + first[6]] == images[0][1]
    assert data[second[7] : second[7] + second[6]] == images[1][1]


def test_dib_32_stores_bottom_up_bgra_and_a_transparency_mask() -> None:
    make_icon = _load_make_icon()
    rgba = bytes(
        [
            255, 0, 0, 255,
            0, 0, 0, 0,
            0, 255, 0, 128,
            0, 0, 255, 255,
        ]
    )
    dib = make_icon.dib_32(2, 2, rgba)
    header = struct.unpack_from("<IiiHHIIiiII", dib, 0)
    assert header[0] == 40
    assert header[1] == 2
    assert header[2] == 4
    assert header[4] == 32
    # Bottom row is stored first: green at partial alpha, then opaque blue.
    assert dib[40:44] == bytes((0, 255, 0, 128))
    assert dib[44:48] == bytes((255, 0, 0, 255))
    assert dib[48:52] == bytes((0, 0, 255, 255))
    assert dib[52:56] == bytes((0, 0, 0, 0))
    # The clear pixel is the second pixel of the top row, which is the second mask byte's high bits.
    mask = dib[56:]
    assert mask[0] == 0
    assert mask[4] == 0b0100_0000


def test_installer_shortcuts_use_the_icon_file() -> None:
    text = Path(__file__).resolve().parents[1].joinpath("packaging", "windows", "Outburst.iss").read_text(
        encoding="utf-8"
    )
    assert 'Source: "..\\..\\build\\windows\\outburst.ico"; DestDir: "{app}"' in text
    assert text.count('IconFilename: "{app}\\outburst.ico"') == 2
    assert "UninstallDisplayIcon={app}\\outburst.ico" in text


def test_icon_file_uses_the_film_strip_artwork(tmp_path: Path) -> None:
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    make_icon = _load_make_icon()
    QApplication.instance() or QApplication([])
    from outburst.gui.styles import _icon_image

    source = _icon_image().convertToFormat(QImage.Format.Format_RGBA8888)
    assert not source.isNull()
    rgba = make_icon.rgba_from_image(source)
    destination = tmp_path / "outburst.ico"
    make_icon.write_ico(destination, [(256, make_icon.dib_32(256, 256, rgba))])
    data = destination.read_bytes()
    _reserved, _kind, count = struct.unpack_from("<HHH", data, 0)
    assert count == 1
    entry = struct.unpack_from("<BBBBHHII", data, 6)
    assert entry[0] == 0
    dib = data[entry[7] : entry[7] + entry[6]]
    # The artwork's top-left corner is transparent, so it is the last bitmap row and the mask marks it.
    corner = dib[40 + (255 * 256 * 4) : 40 + (255 * 256 * 4) + 4]
    assert corner[3] == 0
    mask_stride = ((256 + 31) // 32) * 4
    mask = dib[40 + (256 * 256 * 4) :]
    assert mask[(255 * mask_stride)] & 0b1000_0000
