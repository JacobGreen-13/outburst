"""Keyboard shortcuts shown in the Help menu and the README.

``shortcuts_html`` turns the ``SHORTCUTS`` table into the Help dialog. The window handles the keys itself so a focused slider is left alone.
"""

from __future__ import annotations

SHORTCUTS: tuple[tuple[str, str], ...] = (
    ("Right / Left", "Next or previous frame in the filmstrip order"),
    ("Down / Up", "Next or previous burst"),
    ("Home / End", "First or last frame in the filmstrip order"),
    ("Enter", "Jump to the recommended frame"),
    ("C", "Compare two frames. Starts beside the best other frame in the burst."),
    ("[ / ]", "Move the other side of compare along the filmstrip."),
    ("Shift-click", "Compare the current frame with that thumbnail."),
    ("Z", "Zoom the preview to 1:1. Press again to fit the frame."),
    ("+ / −", "Zoom in or out: fit, 1:1, 2:1, 4:1. The scroll wheel does the same over the preview."),
    ("F", "Focus peaking. Green marks the strongest edges."),
    ("1–5", "Set that star rating. Press the same number again to clear it."),
    ("0", "Clear the star rating."),
    ("A", "Keep the current frame. Same as Keep. Press again to clear it."),
    ("X", "Reject the current frame. The photograph stays where it is, and Save keepers skips it. Press again to clear it."),
    ("R", "Reject every frame in the burst. Press again to clear them."),
    ("G", "Reject obvious misses in this burst. Press again to restore them."),
    ("Ctrl+O", "Open a folder"),
    ("F5", "Clear cached measurements for this folder and scan it again"),
    ("Ctrl+E", "Save the frames marked as keepers"),
    ("Ctrl+Shift+E", "Save a JSON and CSV manifest of every frame, its score, and the keeper marks"),
    ("Ctrl+Shift+X", "Write XMP sidecars for keepers and star-rated frames"),
    ("Esc", "Cancel the scan in progress"),
)


def shortcuts_html() -> str:
    """
    Build the HTML table of keyboard shortcuts.

    Inputs: none.
    Output: an HTML fragment with one row per shortcut and a note about focus.
    Uses: the Help menu shortcuts dialog.
    Limitations: the note says these shortcuts are ignored while a slider or number box has focus, because arrow keys then adjust that control.
    """
    rows = "".join(
        f"<tr><td style='padding:4px 16px 4px 0; font-weight:600;'>{keys}</td><td>{description}</td></tr>"
        for keys, description in SHORTCUTS
    )
    return (
        "<p>These shortcuts work unless a slider or number box has focus. "
        "Arrow keys then adjust that control instead.</p>"
        f"<table>{rows}</table>"
    )
