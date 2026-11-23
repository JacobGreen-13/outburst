"""Copy the Outburst icon to a destination PNG.

The artwork lives at src/outburst/gui/outburst.png. The Debian package installs that file directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "outburst" / "gui" / "outburst.png"


def main() -> None:
    """
    Copy the committed icon.

    Inputs: the destination path in argv.
    Output: no return value. Raises SystemExit when the path is missing or the source cannot be read.
    Uses: a manual refresh of an exported PNG. The package build copies the source itself.
    Limitations: this does not draw a new mark. The destination is a byte copy of the committed PNG.
    """

    if len(sys.argv) != 2:
        raise SystemExit("usage: make_icon.py DESTINATION.png")
    if not SOURCE.is_file():
        raise SystemExit(f"Icon file is missing: {SOURCE}")
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(SOURCE.read_bytes())


if __name__ == "__main__":
    main()
