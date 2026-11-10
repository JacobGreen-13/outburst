"""Run subject and bird eye models in the private Python installed beside a packaged app.

The frozen Windows app cannot import PyTorch. ``python -m outburst.detect_worker`` reads one JSON command per line from stdin and writes one JSON result per line to stdout. ``load`` opens a weights file. ``subject`` and ``eye`` score ``.npy`` BGR previews. ``quit`` stops the process. ``OUTBURST_DETECT_STUB=1`` returns a fixed box and eye without loading Ultralytics.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    """
    Serve detection commands until stdin closes or a quit command arrives.

    Inputs: none. Commands arrive as JSON lines on stdin.
    Output: no return value. Each command writes one JSON line to stdout.
    Uses: the packaged app after subject and bird eye detection are installed.
    Limitations: one command runs at a time. A bad JSON line returns an error and the process keeps reading. Photographs are not opened.
    """

    state = _State()
    for raw in sys.stdin:
        text = raw.strip()
        if not text:
            continue
        command = ""
        try:
            message = json.loads(text)
            command = str(message.get("cmd", ""))
            reply = _handle(state, message)
        except Exception as exc:
            reply = {"ok": False, "error": str(exc)}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()
        if command == "quit":
            return


class _State:
    """Models kept for the life of this process."""

    def __init__(self) -> None:
        """
        Start with no model loaded.

        Inputs: none.
        Output: no return value. Both model slots are empty.
        Uses: main.
        Limitations: this object stays on the one worker thread.
        """

        self.subject: object | None = None
        self.eye: object | None = None


def _handle(state: _State, message: dict[str, object]) -> dict[str, object]:
    """
    Run one detection command.

    Inputs: state, the loaded models; message, one decoded command.
    Output: a JSON-ready dict with ok True, or a dict the caller treats as the reply.
    Uses: main.
    Limitations: an unknown command raises. Stub mode does not load weights.
    """

    command = str(message.get("cmd", ""))
    if command == "quit":
        return {"ok": True}
    if os.environ.get("OUTBURST_DETECT_STUB") == "1":
        return _handle_stub(command, message)
    if command == "load":
        _load(state, message)
        return {"ok": True}
    files = _files(message)
    images = [np.load(path) for path in files]
    if command == "subject":
        if state.subject is None:
            raise RuntimeError("The subject model is not loaded.")
        found = state.subject.detect_many(images)  # type: ignore[attr-defined]
        return {"ok": True, "boxes": [_box_payload(item) for item in found]}
    if command == "eye":
        if state.eye is None:
            raise RuntimeError("The bird eye model is not loaded.")
        found = state.eye.find_many(images)  # type: ignore[attr-defined]
        return {"ok": True, "eyes": [_eye_payload(item) for item in found]}
    raise RuntimeError(f"Unknown detection command: {command}")


def _handle_stub(command: str, message: dict[str, object]) -> dict[str, object]:
    """
    Answer a command without loading a model.

    Inputs: command, the cmd field; message, the decoded command.
    Output: a fixed subject box or eye for each file, or ok for load and quit.
    Uses: tests and a packaged install check that does not download weights.
    Limitations: the box and eye are not measured from the preview. An unknown command raises.
    """

    if command == "load":
        return {"ok": True}
    count = len(_files(message))
    if command == "subject":
        box = {"box": [8, 8, 40, 30], "label": "bird", "confidence": 0.9}
        return {"ok": True, "boxes": [box for _index in range(count)]}
    if command == "eye":
        eye = {"x": 20, "y": 30, "radius": 12, "sharpness": 18.0}
        return {"ok": True, "eyes": [eye for _index in range(count)]}
    raise RuntimeError(f"Unknown detection command: {command}")


def _load(state: _State, message: dict[str, object]) -> None:
    """
    Open the subject model, the bird eye model, or both.

    Inputs: state, the model slots to replace; message, a load command with optional subject and eye paths.
    Output: no return value. A path that is present replaces that slot.
    Uses: the load command.
    Limitations: a missing file raises FileNotFoundError. Loading imports Ultralytics in this process.
    """

    subject = message.get("subject")
    eye = message.get("eye")
    if isinstance(subject, str) and subject:
        state.subject = _open_subject(Path(subject))
    if isinstance(eye, str) and eye:
        state.eye = _open_eye(Path(eye))


def _open_subject(path: Path) -> object:
    """
    Load one subject weights file in this private Python.

    Inputs: path, the .pt file.
    Output: a YoloSubjectDetector.
    Uses: _load.
    Limitations: this imports Ultralytics. The file must already be on disk.
    """

    from ultralytics import YOLO

    from outburst.subject import YoloSubjectDetector, preferred_device

    return YoloSubjectDetector(YOLO(str(path)), preferred_device())


def _open_eye(path: Path) -> object:
    """
    Load one bird eye weights file in this private Python.

    Inputs: path, the .pt file.
    Output: a BirdEyeDetector.
    Uses: _load.
    Limitations: this imports Ultralytics. The pose model stays off MPS.
    """

    from ultralytics import YOLO

    from outburst.eyes import BirdEyeDetector, pose_device

    return BirdEyeDetector(YOLO(str(path)), pose_device())


def _files(message: dict[str, object]) -> list[str]:
    """
    Read the preview paths from a command.

    Inputs: message, a subject or eye command.
    Output: the file path strings. An empty list when files is missing.
    Uses: _handle and the stub.
    Limitations: entries that are not strings are skipped.
    """

    raw = message.get("files")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str)]


def _box_payload(item: object) -> dict[str, object] | None:
    """
    Turn one subject detection into JSON fields.

    Inputs: item, a Detection or None.
    Output: box, label, and confidence, or None.
    Uses: a subject command reply.
    Limitations: the box is four integers. This does not run the model.
    """

    if item is None:
        return None
    box = getattr(item, "box")
    return {
        "box": [int(value) for value in box],
        "label": str(getattr(item, "label")),
        "confidence": float(getattr(item, "confidence")),
    }


def _eye_payload(item: object) -> dict[str, object] | None:
    """
    Turn one eye into JSON fields.

    Inputs: item, an Eye or None.
    Output: x, y, radius, and sharpness, or None.
    Uses: an eye command reply.
    Limitations: sharpness may be null. This does not run the model.
    """

    if item is None:
        return None
    sharpness = getattr(item, "sharpness")
    return {
        "x": int(getattr(item, "x")),
        "y": int(getattr(item, "y")),
        "radius": int(getattr(item, "radius")),
        "sharpness": None if sharpness is None else float(sharpness),
    }


if __name__ == "__main__":
    main()
