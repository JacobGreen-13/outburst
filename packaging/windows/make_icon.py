"""Write the Windows icon used by the frozen app and the installer.

Run this after Outburst and PySide6 are installed. The icon is the same stacked-frame mark the window draws in memory.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path


def write_ico(destination: Path, images: list[tuple[int, bytes]]) -> None:
    """Write a PNG-in-ICO file that Windows 11 can use for the app and the setup program.

    Inputs: destination, the .ico path; images, pairs of pixel size and PNG bytes.
    Output: no return value. The file is created or replaced.
    Uses: the PyInstaller executable and the Inno Setup wizard.
    Limitations: each image is a PNG. Windows Vista and later read that form. The size byte is 0 when the image is 256 pixels, which is the ICO rule for that size.
    """

    count = len(images)
    header = struct.pack("<HHH", 0, 1, count)
    entries = bytearray()
    payload = bytearray()
    offset = 6 + (16 * count)
    for size, png in images:
        dimension = 0 if size >= 256 else size
        entries += struct.pack("<BBBBHHII", dimension, dimension, 0, 0, 1, 32, len(png), offset)
        payload += png
        offset += len(png)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(header + entries + payload)


def main() -> None:
    """Draw the Outburst mark at several sizes and save an .ico.

    Inputs: the destination path in argv, or build/windows/outburst.ico when argv is empty.
    Output: no return value. Raises SystemExit when the path is missing or a PNG cannot be encoded.
    Uses: packaging/windows/build.ps1 before PyInstaller runs.
    Limitations: a QApplication is created so the pixmap can be painted. The process exits after the file is written.
    """

    from PySide6.QtCore import QBuffer, QIODevice
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    destination = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/windows/outburst.ico")
    app = QApplication.instance() or QApplication([])
    del app
    from outburst.gui.styles import build_app_icon

    icon = build_app_icon()
    images: list[tuple[int, bytes]] = []
    for size in (16, 32, 48, 256):
        image = icon.pixmap(size, size).toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        buffer = QBuffer()
        if not buffer.open(QIODevice.OpenModeFlag.WriteOnly):
            raise SystemExit(f"Could not encode the {size}px icon")
        if not image.save(buffer, "PNG"):
            raise SystemExit(f"Could not encode the {size}px icon")
        images.append((size, bytes(buffer.data())))
        buffer.close()
    write_ico(destination, images)


if __name__ == "__main__":
    main()
