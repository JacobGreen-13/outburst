"""SQLite cache for timestamps, measurements, and thumbnail files.

The cache lives in the user cache directory, never beside the photographs.
Rows are keyed by path, size, modification time, score version, and subject
model so a retake or a formula change is scored again and an unchanged file
is not.

``default_cache_dir`` picks the user cache folder. ``Cache.get`` and ``Cache.put`` store measurements and preview paths. ``decisions_for``, ``decision_origins``, and ``set_decision`` store picks and rejects. ``stars_for`` and ``set_stars`` store ratings. ``forget_folder`` and ``clear`` drop measurements and leave decisions and ratings. The remaining helpers migrate columns and resolve preview paths.
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
    critical_sharpness REAL,
    tenengrad REAL,
    exposure REAL,
    highlight_clip REAL,
    shadow_clip REAL,
    subject_sharpness REAL,
    subject_confidence REAL,
    subject_label TEXT,
    subject_box TEXT,
    eye_sharpness REAL,
    eye_x INTEGER,
    eye_y INTEGER,
    eye_radius INTEGER,
    preview_rel TEXT,
    thumb_rel TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (path, mtime_ns, size, score_version, subject_key)
);

CREATE TABLE IF NOT EXISTS decisions (
    path TEXT PRIMARY KEY,
    decision TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'user'
);

CREATE TABLE IF NOT EXISTS ratings (
    path TEXT PRIMARY KEY,
    stars INTEGER NOT NULL,
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
    """Per-user cache directory. Created on first use.

    Inputs: none.
    Output: the platform cache path, created if it is missing.
    Uses: the review window and the scan pipeline when no cache directory is passed.
    Limitations: creates the directory on disk and does not write photographs.
    """

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
        """Open or create the SQLite cache and its preview folders.

        Inputs: the directory that holds outburst.sqlite, previews, and thumbs.
        Output: none; the connection is stored on this instance.
        Uses: the review window and the scan pipeline when they open a cache.
        Limitations: adds missing columns to an older database and does not write photographs.
        """

        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "previews").mkdir(exist_ok=True)
        (self.directory / "thumbs").mkdir(exist_ok=True)
        self._path = self.directory / "outburst.sqlite"
        self._conn = sqlite3.connect(self._path, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._ensure_decision_origin()
        self._ensure_critical_sharpness()
        self._ensure_eye_columns()
        self._conn.commit()

    def close(self) -> None:
        """Close the SQLite connection.

        Inputs: none beyond this cache.
        Output: none.
        Uses: __exit__ when the cache is used as a context manager.
        Limitations: does not delete rows or preview files.
        """

        self._conn.close()

    def __enter__(self) -> Cache:
        """Begin a cache context.

        Inputs: none beyond this cache.
        Output: this Cache instance.
        Uses: callers that open the cache with a with statement.
        Limitations: does not open a second connection.
        """

        return self

    def __exit__(self, *args: object) -> None:
        """Close the cache when the with block ends.

        Inputs: the usual exception triple, which is ignored.
        Output: none, so an exception continues to propagate.
        Uses: the review window and the scan pipeline.
        Limitations: does not roll back a write that was already committed.
        """

        self.close()

    def get(
        self,
        path: Path,
        mtime_ns: int,
        size: int,
        subject_key: str,
    ) -> CacheRecord | None:
        """Load one cached measurement when the file identity still matches.

        Inputs: the photograph path, mtime, size, and subject model key.
        Output: a CacheRecord, or None when the row or its preview files are missing.
        Uses: the scan pipeline to skip work on an unchanged file.
        Limitations: a different score version misses, and this does not write photographs.
        """

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
            critical_sharpness=row["critical_sharpness"],
            subject_sharpness=row["subject_sharpness"],
            subject_box=_parse_box(row["subject_box"]),
            subject_label=row["subject_label"],
            subject_confidence=row["subject_confidence"],
            eye_sharpness=None if row["eye_sharpness"] is None else float(row["eye_sharpness"]),
            eye_center=_parse_center(row["eye_x"], row["eye_y"]),
            eye_radius=None if row["eye_radius"] is None else int(row["eye_radius"]),
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
        """Store one measurement and the relative paths of its preview files.

        Inputs: file identity, capture time, camera, metrics, and preview paths.
        Output: none; the row is inserted or replaced and committed.
        Uses: the scan pipeline after a frame is scored.
        Limitations: stores paths relative to the cache directory and does not copy the photograph.
        """

        self._conn.execute(
            """
            INSERT OR REPLACE INTO frame_metrics (
                path, mtime_ns, size, score_version, subject_key,
                captured_at, timestamp_source, camera,
                sharpness, center_sharpness, critical_sharpness, tenengrad, exposure,
                highlight_clip, shadow_clip, subject_sharpness, subject_confidence,
                subject_label, subject_box, eye_sharpness, eye_x, eye_y, eye_radius,
                preview_rel, thumb_rel, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                metrics.critical_sharpness,
                metrics.tenengrad,
                metrics.exposure,
                metrics.highlight_clip,
                metrics.shadow_clip,
                metrics.subject_sharpness,
                metrics.subject_confidence,
                metrics.subject_label,
                _format_box(metrics.subject_box),
                metrics.eye_sharpness,
                None if metrics.eye_center is None else metrics.eye_center[0],
                None if metrics.eye_center is None else metrics.eye_center[1],
                metrics.eye_radius,
                self._rel(preview_path),
                self._rel(thumb_path),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        self._conn.commit()

    def decisions_for(self, paths: list[Path]) -> dict[Path, str]:
        """Read keep and reject marks for the given photographs.

        Inputs: photograph paths to look up.
        Output: a path-to-decision map containing only keep and reject.
        Uses: the review window when a folder is opened.
        Limitations: an empty path list returns an empty dict, and other decision text is skipped.
        """

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

    def decision_origins(self, paths: list[Path]) -> dict[Path, str]:
        """``auto`` when Outburst marked the keeper, ``user`` when the photographer did.

        Inputs: photograph paths to look up.
        Output: a path-to-origin map of user or auto.
        Uses: the review window so an automatic keeper can be replaced.
        Limitations: an unrecognized origin is returned as user.
        """

        found: dict[Path, str] = {}
        if not paths:
            return found
        for start in range(0, len(paths), 400):
            chunk = paths[start : start + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT path, origin FROM decisions WHERE path IN ({placeholders})",
                [str(path) for path in chunk],
            )
            by_text = {str(path): path for path in chunk}
            for row in rows:
                path = by_text.get(row["path"])
                if path is None:
                    continue
                found[path] = row["origin"] if row["origin"] in {"user", "auto"} else "user"
        return found

    def set_decision(self, path: Path, decision: str | None, *, origin: str = "user") -> None:
        """Save or clear one keep or reject mark.

        Inputs: a photograph path, keep, reject, or None, and an origin of user or auto.
        Output: none; the row is replaced or deleted and committed.
        Uses: the review window when the photographer flags a frame.
        Limitations: a reject is stored as user origin, and None deletes the mark.
        """

        if decision not in {None, "keep", "reject"}:
            raise ValueError(f"Unknown decision: {decision}")
        if origin not in {"user", "auto"}:
            raise ValueError(f"Unknown decision origin: {origin}")
        if decision is None:
            self._conn.execute("DELETE FROM decisions WHERE path = ?", (str(path),))
        else:
            stored_origin = origin if decision == "keep" else "user"
            self._conn.execute(
                """
                INSERT OR REPLACE INTO decisions (path, decision, updated_at, origin)
                VALUES (?, ?, ?, ?)
                """,
                (
                    str(path),
                    decision,
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    stored_origin,
                ),
            )
        self._conn.commit()

    def stars_for(self, paths: list[Path]) -> dict[Path, int]:
        """Star ratings the photographer set. Missing paths have no rating yet.

        Inputs: photograph paths to look up.
        Output: a path-to-stars map for ratings from 0 through 5.
        Uses: the review window when a folder is opened.
        Limitations: ratings outside 0 through 5 are omitted.
        """

        found: dict[Path, int] = {}
        if not paths:
            return found
        for start in range(0, len(paths), 400):
            chunk = paths[start : start + 400]
            placeholders = ",".join("?" for _ in chunk)
            rows = self._conn.execute(
                f"SELECT path, stars FROM ratings WHERE path IN ({placeholders})",
                [str(path) for path in chunk],
            )
            by_text = {str(path): path for path in chunk}
            for row in rows:
                path = by_text.get(row["path"])
                stars = row["stars"]
                if path is not None and isinstance(stars, int) and 0 <= stars <= 5:
                    found[path] = stars
        return found

    def set_stars(self, path: Path, stars: int | None) -> None:
        """Save or clear one star rating.

        Inputs: a photograph path and a rating from 0 through 5, or None.
        Output: none; the row is replaced or deleted and committed.
        Uses: the review window when the photographer rates a frame.
        Limitations: None deletes the rating, and a value outside 0 through 5 raises ValueError.
        """

        if stars is not None and stars not in range(6):
            raise ValueError(f"Star rating must be 0 through 5, not {stars}")
        if stars is None:
            self._conn.execute("DELETE FROM ratings WHERE path = ?", (str(path),))
        else:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO ratings (path, stars, updated_at)
                VALUES (?, ?, ?)
                """,
                (str(path), stars, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
        self._conn.commit()

    def _ensure_decision_origin(self) -> None:
        """Add the origin column to an older decisions table.

        Inputs: none beyond this cache.
        Output: none.
        Uses: Cache initialization before the first commit.
        Limitations: a missing column defaults existing rows to user and does not rewrite marks.
        """

        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(decisions)")}
        if "origin" in columns:
            return
        self._conn.execute(
            "ALTER TABLE decisions ADD COLUMN origin TEXT NOT NULL DEFAULT 'user'"
        )

    def _ensure_critical_sharpness(self) -> None:
        """Add the critical sharpness column to an older metrics table.

        Inputs: none beyond this cache.
        Output: none.
        Uses: Cache initialization before the first commit.
        Limitations: older rows keep a null critical sharpness until they are scored again.
        """

        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(frame_metrics)")}
        if "critical_sharpness" in columns:
            return
        self._conn.execute("ALTER TABLE frame_metrics ADD COLUMN critical_sharpness REAL")

    def _ensure_eye_columns(self) -> None:
        """Add eye measurement columns to an older metrics table.

        Inputs: none beyond this cache.
        Output: none.
        Uses: Cache initialization before the first commit.
        Limitations: older rows keep null eye fields until they are scored again.
        """

        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(frame_metrics)")}
        additions = (
            ("eye_sharpness", "REAL"),
            ("eye_x", "INTEGER"),
            ("eye_y", "INTEGER"),
            ("eye_radius", "INTEGER"),
        )
        for name, kind in additions:
            if name not in columns:
                self._conn.execute(f"ALTER TABLE frame_metrics ADD COLUMN {name} {kind}")

    def forget_folder(self, folder: Path) -> int:
        """Drop cached measurements and preview files for photographs inside ``folder``.

        Keeper and reject marks are left in place. Returns how many photograph
        paths were forgotten.

        Inputs: a folder path, which is expanded and resolved.
        Output: the number of photograph paths whose measurements were removed.
        Uses: the review window when the photographer forgets one folder.
        Limitations: keeper, reject, and star rows stay in the database.
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

        Inputs: none beyond this cache.
        Output: the number of measurement rows removed.
        Uses: the review window when the photographer clears the cache.
        Limitations: keeper, reject, and star rows stay in the database.
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
        """Delete one preview or thumbnail if it sits inside the cache directory.

        Inputs: a path relative to the cache directory, or None.
        Output: none.
        Uses: forget_folder and clear.
        Limitations: a missing file or a path outside the cache is left alone, and a failed unlink is logged.
        """

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
        """Delete leftover files in the preview and thumbnail folders.

        Inputs: none beyond this cache.
        Output: none.
        Uses: clear after measurement rows are removed.
        Limitations: only regular files are removed, and a failed unlink is logged.
        """

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
        """Turn an absolute cache path into a posix path relative to the cache directory.

        Inputs: a path that already lives under the cache directory.
        Output: a relative posix path string.
        Uses: put when storing preview and thumbnail locations.
        Limitations: a path outside the cache directory raises ValueError.
        """

        return path.resolve().relative_to(self.directory.resolve()).as_posix()

    def _abs(self, relative: str | None) -> Path | None:
        """Turn a stored relative path back into a path under the cache directory.

        Inputs: a relative path string, or None.
        Output: a path under the cache directory, or None when the stored value is empty.
        Uses: get and _delete_cached_file.
        Limitations: does not check that the file exists.
        """

        if not relative:
            return None
        return self.directory / relative


def _stored_path_inside(stored: str, folder: Path) -> bool:
    """Tell whether a stored photograph path is the folder or a child of it.

    Inputs: a path string from the database and a resolved folder.
    Output: True when the path is the folder or lies under it.
    Uses: forget_folder while scanning measurement rows.
    Limitations: an OSError during resolve falls back to the stored path as written.
    """

    path = Path(stored)
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path
    return resolved == folder or folder in resolved.parents


def _format_box(box: tuple[int, int, int, int] | None) -> str | None:
    """Turn a subject box into a comma-separated string for SQLite.

    Inputs: x, y, width, and height, or None.
    Output: a four-integer string, or None.
    Uses: put when storing metrics.subject_box.
    Limitations: None stays None, and each part is stored as an integer.
    """

    if box is None:
        return None
    return ",".join(str(int(part)) for part in box)


def _parse_box(value: str | None) -> tuple[int, int, int, int] | None:
    """Read a stored subject box back into four integers.

    Inputs: a comma-separated string, or None.
    Output: a four-integer box, or None when the text is empty or not four integers.
    Uses: get when rebuilding Metrics.
    Limitations: a malformed value returns None instead of raising.
    """

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


def _parse_center(x: object, y: object) -> tuple[int, int] | None:
    """Read stored eye coordinates into a pair of integers.

    Inputs: the eye_x and eye_y column values.
    Output: an integer pair, or None when either value is None.
    Uses: get when rebuilding Metrics.eye_center.
    Limitations: a non-numeric value raises rather than returning None.
    """

    if x is None or y is None:
        return None
    return (int(x), int(y))
