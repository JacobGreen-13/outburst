"""Keyboard shortcuts shown in the Help menu and the README."""

from __future__ import annotations

SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("Right / Left", "Next or previous frame"),
    ("Down / Up", "Next or previous burst"),
    ("Home / End", "First or last frame in the burst"),
    ("Enter", "Jump to the recommended frame"),
    ("A", "Mark the current frame as a keeper. Press again to clear it."),
    ("X", "Reject the current frame. Press again to clear it."),
    ("R", "Reject every frame in the burst. Press again to clear them."),
    ("Ctrl+O", "Open a folder"),
    ("F5", "Clear cached measurements for this folder and scan it again"),
    ("Ctrl+E", "Save the frames marked as keepers"),
    ("Ctrl+Shift+E", "Export a JSON and CSV report"),
    ("Ctrl+Shift+X", "Write XMP sidecars"),
    ("Esc", "Cancel the scan in progress"),
)


def shortcuts_html() -> str:
    rows = "".join(
        f"<tr><td style='padding:4px 16px 4px 0; font-weight:600;'>{keys}</td><td>{description}</td></tr>"
        for keys, description in SHORTCUTS
    )
    return (
        "<p>These shortcuts work unless a slider or number box has focus. "
        "Arrow keys then adjust that control instead.</p>"
        f"<table>{rows}</table>"
    )
