"""Keep Qt offscreen when the GUI tests run.

This file defines no functions. On import it sets ``QT_QPA_PLATFORM`` to offscreen and ``OUTBURST_HEADLESS`` to 1 when those variables are unset.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("OUTBURST_HEADLESS", "1")
