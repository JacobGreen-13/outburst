"""Launch the Outburst review window.

``main`` parses an optional folder and ``--version``. ``_run_gui`` applies the theme, shows ``MainWindow``, and opens that folder when one was given. A missing PySide6 install returns status 1. Running this file directly, which is how the Windows installer starts, calls ``main``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from outburst import __version__
from outburst.cache import default_cache_dir
from outburst.logsetup import configure_logging


def main(argv: list[str] | None = None) -> int:
    """
    Parse the command line and open the review window.

    Inputs: argv, the argument list, or None to read the process arguments.
    Output: the Qt event-loop exit code, or 1 when PySide6 cannot be imported.
    Uses: the application entry point that shows the review window.
    Limitations: a folder argument is optional; without one the window opens empty. If preparing the cache directory raises OSError, logging is configured with no cache directory.
    """
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
    """
    Create the Qt application, show the review window, and run the event loop.

    Inputs: folder, a burst folder to open, or None.
    Output: the integer exit code from the Qt event loop, or 1 when PySide6 cannot be imported.
    Uses: review-window startup.
    Limitations: an existing QApplication is reused. The font size is raised to 12 only when the current point size is already positive. A folder is opened after the window is shown. The Debian package sets the desktop file name so the window groups with its launcher.
    """
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
    if os.environ.get("OUTBURST_PACKAGED") == "1":
        app.setDesktopFileName("outburst")
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


if __name__ == "__main__":
    raise SystemExit(main())
