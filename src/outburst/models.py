"""Data objects shared by the culling pipeline and the review window."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Frame:
    """One photograph on disk. RAW files stay the canonical path when a JPEG is paired."""

    path: Path
    kind: str
    captured_at: datetime | None
    timestamp_source: str
    camera: str | None
    size: int
    mtime_ns: int
    paired_jpeg: Path | None = None


@dataclass(frozen=True)
class Burst:
    """Consecutive frames whose capture times fall inside the grouping threshold."""

    id: str
    frames: tuple[Frame, ...]

    @property
    def count(self) -> int:
        return len(self.frames)

    @property
    def started_at(self) -> datetime | None:
        if not self.frames:
            return None
        return self.frames[0].captured_at


@dataclass(frozen=True)
class Metrics:
    """Measurements taken from a preview. Weights are applied later, so these can be cached."""

    sharpness: float
    center_sharpness: float
    tenengrad: float
    exposure: float
    highlight_clip: float
    shadow_clip: float
    critical_sharpness: float | None = None
    subject_sharpness: float | None = None
    subject_box: tuple[int, int, int, int] | None = None
    subject_label: str | None = None
    subject_confidence: float | None = None


@dataclass(frozen=True)
class Weights:
    """How measurements turn into one score. Sharpness is dominant by default."""

    sharpness: float = 0.75
    exposure: float = 0.25
    center_mix: float = 0.65
    prefer_subject: bool = False

    def clamped(self) -> Weights:
        sharpness = max(0.0, float(self.sharpness))
        exposure = max(0.0, float(self.exposure))
        if sharpness + exposure == 0:
            sharpness, exposure = 0.75, 0.25
        center_mix = min(1.0, max(0.0, float(self.center_mix)))
        return Weights(sharpness, exposure, center_mix, bool(self.prefer_subject))


@dataclass(frozen=True)
class FrameAnalysis:
    """A frame plus the preview files and score used by the review window."""

    frame: Frame
    metrics: Metrics | None
    composite: float | None
    preview_path: Path | None
    thumb_path: Path | None
    error: str | None


@dataclass(frozen=True)
class Library:
    """Everything the window needs after a folder has been scanned."""

    folder: Path
    analyses: tuple[FrameAnalysis, ...]
    bursts: tuple[Burst, ...]
    warnings: tuple[str, ...]
    threshold: float
    weights: Weights
    subject_active: bool = False
