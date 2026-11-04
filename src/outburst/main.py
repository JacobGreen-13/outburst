"""Launch the Outburst review window."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from outburst import __version__
from outburst.cache import default_cache_dir
from outburst.logsetup import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="outburst",
        description="Outburst — local, offline burst culling for wildlife and nature photographers.",
    )
    parser.add_argument("folder", nargs="?", type=Path, help="Folder of RAW or JPEG bursts to open")
    parser.add_argument("--version", action="version", version=f"Outburst {__version__}")
    args = parser.parse_args(argv)
    try:
        configure_logging(default_cache_dir())
    except OSError:
        configure_logging(None)
    return _run_gui(args.folder)


def _run_gui(folder: Path | None) -> int:
    try:
        from PySide6.QtGui import QFont
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        print("Outburst needs PySide6. Install this project with: pip install -e .", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 1

    from outburst.gui.main_window import MainWindow
    from outburst.gui.styles import apply_theme, build_app_icon

    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Outburst")
    app.setOrganizationName("Outburst")
    app.setApplicationDisplayName("Outburst")
    font = QFont(app.font())
    if font.pointSize() > 0:
        font.setPointSize(max(font.pointSize(), 12))
    app.setFont(font)
    apply_theme(app)
    app.setWindowIcon(build_app_icon())
    window = MainWindow()
    window.show()
    if folder is not None:
        window.open_path(folder)
    return int(app.exec())
