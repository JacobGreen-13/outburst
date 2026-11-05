"""Find RAW and JPEG photographs in a folder without touching their contents."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

RAW_EXTENSIONS = frozenset(
    {
        ".3fr",
        ".arw",
        ".cap",
        ".cr2",
        ".cr3",
        ".dcr",
        ".dng",
        ".erf",
        ".fff",
        ".gpr",
        ".iiq",
        ".kdc",
        ".mef",
        ".mos",
        ".mrw",
        ".nef",
        ".nrw",
        ".orf",
        ".pef",
        ".raf",
        ".raw",
        ".rw2",
        ".rwl",
        ".sr2",
        ".srf",
        ".srw",
        ".x3f",
    }
)
JPEG_EXTENSIONS = frozenset({".jpg", ".jpeg"})
SUPPORTED_EXTENSIONS = RAW_EXTENSIONS | JPEG_EXTENSIONS


@dataclass(frozen=True)
class ScannedFile:
    """A photograph discovered on disk. `paired_jpeg` is set when a same-stem JPEG sits beside a RAW."""

    path: Path
    kind: str
    paired_jpeg: Path | None
    size: int
    mtime_ns: int


def scan_folder(folder: Path, *, recursive: bool = False) -> list[ScannedFile]:
    """Return RAW files, plus JPEGs that are not sitting beside a same-named RAW.

    Inputs: a folder and whether to walk subfolders.
    Output: scanned photographs sorted by name.
    Uses: the scan pipeline when a folder is opened.
    Limitations: a missing folder raises NotADirectoryError, and file contents are not read.
    """

    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {folder}")

    raws: list[Path] = []
    jpegs: dict[tuple[Path, str], Path] = {}
    for path in _iter_files(folder, recursive):
        extension = path.suffix.lower()
        if extension in RAW_EXTENSIONS:
            raws.append(path)
        elif extension in JPEG_EXTENSIONS:
            jpegs[(path.parent, path.stem.lower())] = path

    used_jpegs: set[Path] = set()
    found: list[ScannedFile] = []
    for raw in raws:
        pair = jpegs.get((raw.parent, raw.stem.lower()))
        if pair is not None:
            used_jpegs.add(pair)
        scanned = _describe(raw, "raw", pair)
        if scanned is not None:
            found.append(scanned)

    for jpeg in jpegs.values():
        if jpeg in used_jpegs:
            continue
        scanned = _describe(jpeg, "jpeg", None)
        if scanned is not None:
            found.append(scanned)

    found.sort(key=lambda item: (item.path.name.lower(), str(item.path)))
    return found


def _describe(path: Path, kind: str, paired_jpeg: Path | None) -> ScannedFile | None:
    """Stat one photograph into a ScannedFile.

    Inputs: a path, a kind of raw or jpeg, and an optional paired JPEG.
    Output: a ScannedFile, or None when the file cannot be stat'd.
    Uses: scan_folder for each RAW and leftover JPEG.
    Limitations: an OSError is logged and that file is skipped.
    """

    try:
        stat = path.stat()
    except OSError as exc:
        log.warning("Skipping %s: %s", path, exc)
        return None
    return ScannedFile(
        path=path,
        kind=kind,
        paired_jpeg=paired_jpeg,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def _iter_files(folder: Path, recursive: bool) -> list[Path]:
    """List photograph candidates in a folder.

    Inputs: a folder and whether to walk subfolders.
    Output: file paths, excluding names that start with a dot.
    Uses: scan_folder before extensions are filtered.
    Limitations: a folder that cannot be listed is logged and skipped, and symbolic links are not followed.
    """

    files: list[Path] = []

    def on_error(error: OSError) -> None:
        """Log a directory walk error and continue.

        Inputs: the OSError from os.walk.
        Output: none.
        Uses: os.walk when a subdirectory cannot be listed.
        Limitations: the error is only logged, and the walk continues.
        """

        log.warning("Could not list a folder: %s", error)

    if not recursive:
        try:
            entries = list(folder.iterdir())
        except OSError as exc:
            log.warning("Could not list %s: %s", folder, exc)
            return files
        for entry in entries:
            if entry.name.startswith(".") or not entry.is_file():
                continue
            files.append(entry)
        return files

    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False, onerror=on_error):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        for name in filenames:
            if name.startswith("."):
                continue
            files.append(Path(dirpath) / name)
    return files
