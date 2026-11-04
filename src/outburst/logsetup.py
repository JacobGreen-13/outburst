"""Logging for the desktop app. Library code only emits records."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_configured = False


def configure_logging(cache_dir: Path | None = None) -> None:
    """Send logs to stderr and, when possible, a rotating file in the cache directory."""

    global _configured
    if _configured:
        return
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    root.addHandler(stream)
    if cache_dir is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                cache_dir / "outburst.log",
                maxBytes=1_000_000,
                backupCount=2,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            root.warning("Could not open the Outburst log file: %s", exc)
    _configured = True
