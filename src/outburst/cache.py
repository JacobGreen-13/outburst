"""SQLite cache for timestamps, measurements, and thumbnail files.

The cache lives in the user cache directory, never beside the photographs.
Rows are keyed by path, size, modification time, score version, and subject
model so a retake or a formula change is scored again and an unchanged file
is not.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from outburst.models import Metrics
from outburst.scoring import SCORE_VERSION

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frame_metrics (
    path TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    score_version INTEGER NOT NULL,
    subject_key TEXT NOT NULL,
    captured_at TEXT,
    timestamp_source TEXT,
    camera TEXT,
    sharpness REAL,
    center_sharpness REAL,
    tenengrad REAL,
    exposure REAL,
    highlight_clip REAL,
    shadow_clip REAL,
    subject_sharpness REAL,
    subject_confidence REAL,
    subject_label TEXT,
    subject_box TEXT,
    preview_rel TEXT,
    thumb_rel TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (path, mtime_ns, size, score_version, subject_key)
);

CREATE TABLE IF NOT EXISTS decisions (
    path TEXT PRIMARY KEY,
    decision TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class CacheRecord:
    captured_at: datetime | None
    timestamp_source: str
    camera: str | None
    metrics: Metrics
    preview_path: Path
    thumb_path: Path


def default_cache_dir() -> Path:
    """Per-user cache directory. Created on first use."""

    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches" / "Outburst"
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        root = base / "Outburst" / "Cache"
    else:
        xdg = os.environ.get("XDG_CACHE_HOME")
        root = Path(xdg) if xdg else Path.home() / ".cache"
        root = root / "outburst"
    root.mkdir(parents=True, exist_ok=True)
    return root


class Cache:
    """One SQLite file plus preview and thumbnail directories."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "previews").mkdir(exist_ok=True)
        (self.directory / "thumbs").mkdir(exist_ok=True)
        self._path = self.directory / "outburst.sqlite"
        self._conn = sqlite3.connect(self._path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Cache:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get(
        self,
        path: Path,
        mtime_ns: int,
        size: int,
        subject_key: str,
    ) -> CacheRecord | None:
        row = self._conn.execute(
            """
            SELECT * FROM frame_metrics
            WHERE path = ? AND mtime_ns = ? AND size = ? AND score_version = ? AND subject_key = ?
            """,
            (str(path), mtime_ns, size, SCORE_VERSION, subject_key),
        ).fetchone()
        if row is None:
            return None
        preview = self._abs(row["preview_rel"])
        thumb = self._abs(row["thumb_rel"])
        if preview is None or thumb is None or not preview.is_file() or not thumb.is_file():
            return None
        metrics = Metrics(
            sharpness=row["sharpness"],
            center_sharpness=row["center_sharpness"],
            tenengrad=row["tenengrad"],
            exposure=row["exposure"],
            highlight_clip=row["highlight_clip"],
            shadow_clip=row["shadow_clip"],
            subject_sharpness=row["subject_sharpness"],
            subject_box=_parse_box(row["subject_box"]),
            subject_label=row["subject_label"],
            subject_confidence=row["subject_confidence"],
        )
        captured = row["captured_at"]
        return CacheRecord(
            captured_at=datetime.fromisoformat(captured) if captured else None,
            timestamp_source=row["timestamp_source"] or "missing",
            camera=row["camera"],
            metrics=metrics,
            preview_path=preview,
            thumb_path=thumb,
        )

    def put(
        self,
        path: Path,
        mtime_ns: int,
        size: int,
        subject_key: str,
        *,
        captured_at: datetime | None,
        timestamp_source: str,
        camera: str | None,
        metrics: Metrics,
        preview_path: Path,
        thumb_path: Path,
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO frame_metrics (
                path, mtime_ns, size, score_version, subject_key,
                captured_at, timestamp_source, camera,
                sharpness, center_sharpness, tenengrad, exposure,
                highlight_clip, shadow_clip, subject_sharpness, subject_confidence,
                subject_label, subject_box, preview_rel, thumb_rel, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(path),
                mtime_ns,
                size,
                SCORE_VERSION,
                subject_key,
                captured_at.isoformat(timespec="microseconds") if captured_at else None,
                timestamp_source,
                camera,
                metrics.sharpness,
                metrics.center_sharpness,
                metrics.tenengrad,
                metrics.exposure,
                metrics.highlight_clip,
                metrics.shadow_clip,
                metrics.subject_sharpness,
                metrics.subject_confidence,
                metrics.subject_label,
                _format_box(metrics.subject_box),
                self._rel(preview_path),
                self._rel(thumb_path),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()

    def decisions_for(self, paths: list[Path]) -> dict[Path, str]:
        found: dict[Path, str] = {}
        if not paths:
            return found
        for start in range(0, len(paths), 400):
            chunk = paths[start : start + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT path, decision FROM decisions WHERE path IN ({placeholders})",
                [str(path) for path in chunk],
            )
            by_text = {str(path): path for path in chunk}
            for row in rows:
                path = by_text.get(row["path"])
                if path is not None and row["decision"] in {"keep", "reject"}:
                    found[path] = row["decision"]
        return found

    def set_decision(self, path: Path, decision: str | None) -> None:
        if decision not in {None, "keep", "reject"}:
            raise ValueError(f"Unknown decision: {decision}")
        if decision is None:
            self._conn.execute("DELETE FROM decisions WHERE path = ?", (str(path),))
        else:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO decisions (path, decision, updated_at)
                VALUES (?, ?, ?)
                """,
                (str(path), decision, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
        self._conn.commit()

    def forget_folder(self, folder: Path) -> int:
        """Drop cached measurements and preview files for photographs inside ``folder``.

        Keeper and reject marks are left in place. Returns how many photograph
        paths were forgotten.
        """

        folder = folder.expanduser().resolve()
        rows = self._conn.execute(
            "SELECT path, preview_rel, thumb_rel FROM frame_metrics"
        ).fetchall()
        paths: set[str] = set()
        for row in rows:
            if not _stored_path_inside(row["path"], folder):
                continue
            paths.add(row["path"])
            self._delete_cached_file(row["preview_rel"])
            self._delete_cached_file(row["thumb_rel"])
        if not paths:
            return 0
        self._conn.executemany(
            "DELETE FROM frame_metrics WHERE path = ?",
            [(path,) for path in paths],
        )
        self._conn.commit()
        return len(paths)

    def clear(self) -> int:
        """Drop every cached measurement and preview file.

        Keeper and reject marks stay. Returns how many measurement rows were removed.
        """

        rows = self._conn.execute("SELECT preview_rel, thumb_rel FROM frame_metrics").fetchall()
        for row in rows:
            self._delete_cached_file(row["preview_rel"])
            self._delete_cached_file(row["thumb_rel"])
        self._conn.execute("DELETE FROM frame_metrics")
        self._conn.commit()
        self._sweep_preview_files()
        return len(rows)

    def _delete_cached_file(self, relative: str | None) -> None:
        path = self._abs(relative)
        if path is None:
            return
        try:
            resolved = path.resolve()
            root = self.directory.resolve()
        except OSError:
            return
        if resolved != root and root not in resolved.parents:
            return
        if not resolved.is_file():
            return
        try:
            resolved.unlink()
        except OSError as exc:
            log.warning("Could not delete cached file %s: %s", resolved, exc)

    def _sweep_preview_files(self) -> None:
        for name in ("previews", "thumbs"):
            folder = self.directory / name
            if not folder.is_dir():
                continue
            for child in folder.iterdir():
                if child.is_file():
                    try:
                        child.unlink()
                    except OSError as exc:
                        log.warning("Could not delete cached file %s: %s", child, exc)

    def _rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.directory.resolve()).as_posix()

    def _abs(self, relative: str | None) -> Path | None:
        if not relative:
            return None
        return self.directory / relative


def _stored_path_inside(stored: str, folder: Path) -> bool:
    path = Path(stored)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return resolved == folder or folder in resolved.parents


def _format_box(box: tuple[int, int, int, int] | None) -> str | None:
    if box is None:
        return None
    return ",".join(str(int(part)) for part in box)


def _parse_box(value: str | None) -> tuple[int, int, int, int] | None:
    if not value:
        return None
    parts = value.split(",")
    if len(parts) != 4:
        return None
    try:
        x, y, width, height = (int(part) for part in parts)
    except ValueError:
        return None
    return (x, y, width, height)
