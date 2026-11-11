"""Write the PNG icon installed by the Debian package.

Run this with the Python that has PySide6. The mark is the same stacked-frame drawing the window uses.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    """
    Draw the Outburst mark and save a PNG.

    Inputs: the destination path in argv.
    Output: no return value. Raises SystemExit when the path is missing or the PNG cannot be written.
    Uses: packaging/linux/build.sh after PySide6 is installed into the package prefix.
    Limitations: Qt runs on the offscreen platform so the build does not need a display. The image is 256 pixels.
    """

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    if len(sys.argv) != 2:
        raise SystemExit("usage: make_icon.py DESTINATION.png")
    destination = Path(sys.argv[1])
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    del app
    from outburst.gui.styles import build_app_icon

    image = build_app_icon().pixmap(256, 256).toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(destination), "PNG"):
        raise SystemExit(f"Could not write {destination}")


if __name__ == "__main__":
    main()
