"""Keep Qt offscreen when the GUI tests run."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OUTBURST_HEADLESS", "1")
