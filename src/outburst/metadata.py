"""Read capture timestamps without modifying the photograph.

exiftool is used when it is installed because it understands the most camera
formats, including CR3. Otherwise Outburst tries the exifread library, then
parses TIFF-based EXIF and JPEG APP1 segments itself, then looks for a date
string in the file header.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import struct
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

SOURCE_EXIFTOOL = "exiftool"
SOURCE_EXIFREAD = "exifread"
SOURCE_EXIF = "exif"
SOURCE_HEADER = "header-scan"
SOURCE_MISSING = "missing"

# exifread recognizes TIFF, JPEG, PNG, WebP, HEIC, and JPEG XL.
# These RAW types use a different container, so exifread does not read them.
_EXIFREAD_UNRECOGNIZED_RAW = frozenset({".cr3", ".raf", ".x3f"})

_EXIF_IFD = 0x8769
_DATETIME = 0x0132
_DATETIME_ORIGINAL = 0x9003
_DATETIME_DIGITIZED = 0x9004
_SUBSEC_ORIGINAL = 0x9291
_SUBSEC = 0x9290
_SUBSEC_DIGITIZED = 0x9292
_MODEL = 0x0110
_MAKE = 0x010F
_INTERESTING_ASCII = {
    _DATETIME,
    _DATETIME_ORIGINAL,
    _DATETIME_DIGITIZED,
    _SUBSEC_ORIGINAL,
    _SUBSEC,
    _SUBSEC_DIGITIZED,
    _MODEL,
    _MAKE,
}
# Standard TIFF, Olympus ORF ("IIRO"), Panasonic RW2.
_TIFF_MAGICS = {42, 0x4F52, 0x0055}
_DATE_RE = re.compile(rb"(?:19|20)\d{2}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}")
_EXIF_MARK = b"Exif\x00\x00"


@dataclass(frozen=True)
class CaptureInfo:
    captured_at: datetime | None
    source: str
    camera: str | None = None


def read_capture_info(paths: list[Path]) -> dict[Path, CaptureInfo]:
    """Return capture metadata for each path. Missing clocks use source `missing`."""

    results: dict[Path, CaptureInfo] = {}
    exiftool_used = False
    exiftool = shutil.which("exiftool")
    if exiftool and paths:
        try:
            results.update(_read_with_exiftool(exiftool, paths))
            exiftool_used = True
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, UnicodeError) as exc:
            log.warning("exiftool failed (%s); trying exifread, then the built-in reader", exc)
            results.clear()

    if paths and not exiftool_used:
        if exifread_available():
            log.info("Reading capture times with exifread because exiftool is not installed")
            for path in paths:
                info = _read_with_exifread(path)
                if info is not None and info.captured_at is not None:
                    results[path] = info
        else:
            log.info("Reading capture times with the built-in reader because exiftool is not installed")

    for path in paths:
        current = results.get(path)
        if current is not None and current.captured_at is not None:
            continue
        builtin = _read_with_builtin(path)
        if builtin.captured_at is not None:
            camera = builtin.camera or (current.camera if current is not None else None)
            results[path] = CaptureInfo(builtin.captured_at, builtin.source, camera)
        elif current is None:
            results[path] = CaptureInfo(None, SOURCE_MISSING, builtin.camera)
    for path in paths:
        results.setdefault(path, CaptureInfo(None, SOURCE_MISSING, None))
    return results


def exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def exifread_available() -> bool:
    try:
        import exifread  # noqa: F401
    except ImportError:
        return False
    return True


def timestamp_reader_warning(raw_suffixes: Iterable[str]) -> str | None:
    """Scan note when RAW files are opened without exiftool. None when exiftool is present."""

    suffixes = {_suffix(value) for value in raw_suffixes}
    suffixes.discard("")
    if not suffixes or exiftool_available():
        return None
    if exifread_available():
        text = (
            "exiftool is not installed, so Outburst is reading capture times with exifread. "
            "exifread covers JPEG and TIFF-based RAW such as CR2, NEF, ARW, DNG, ORF, PEF, and RW2. "
            "CR3, RAF, and X3F are outside that set. Outburst then searches the file for a date string, "
            "which is less reliable, and a frame with no time stays in its own burst. "
            "Install exiftool for CR3 and the less common RAW formats."
        )
        weak = sorted(suffix.lstrip(".").upper() for suffix in suffixes if suffix in _EXIFREAD_UNRECOGNIZED_RAW)
        if weak:
            text += f" This folder includes {', '.join(weak)}."
        return text
    return (
        "exiftool is not installed and exifread is not available, so Outburst is using its built-in "
        "timestamp reader. That reader covers JPEG and many TIFF-based RAW files, then searches for a "
        "date string. CR3 and other non-TIFF RAW files are the least reliable, and a frame with no time "
        "stays in its own burst. Install exiftool for the most reliable burst grouping."
    )


def parse_exif_datetime(value: object, subsec: object | None = None) -> datetime | None:
    """Parse an EXIF datetime, optionally with a separate subsecond field."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or text.startswith("0000"):
        return None
    match = re.match(
        r"(\d{4})[:\-.](\d{2})[:\-.](\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?",
        text,
    )
    if match is None:
        return None
    year, month, day, hour, minute, second, fraction = match.groups()
    try:
        moment = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
    except ValueError:
        return None
    if fraction and (subsec is None or str(subsec).strip() == ""):
        subsec = fraction
    return _apply_subsec(moment, None if subsec is None else str(subsec))


def build_exif_app1(
    when: datetime,
    subsec: str | None = None,
    model: str | None = None,
    *,
    endian: str = "<",
    magic: int = 42,
) -> bytes:
    """Build a JPEG APP1 EXIF segment. Used by tests and the sample-burst generator."""

    tiff = _pack_tiff(when, subsec, model, endian=endian, magic=magic)
    payload = _EXIF_MARK + tiff
    return b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload


def inject_exif(
    jpeg: bytes,
    when: datetime,
    subsec: str | None = None,
    model: str | None = None,
) -> bytes:
    """Insert an EXIF segment after the JPEG start marker."""

    if not jpeg.startswith(b"\xff\xd8"):
        raise ValueError("not a JPEG")
    segment = build_exif_app1(when, subsec, model)
    return jpeg[:2] + segment + jpeg[2:]


def _read_with_exiftool(exiftool: str, paths: list[Path]) -> dict[Path, CaptureInfo]:
    found: dict[Path, CaptureInfo] = {}
    unique_names = len({path.name for path in paths}) == len(paths)
    index: dict[str, Path] = {}
    for path in paths:
        index[str(path)] = path
        index[str(path.resolve())] = path
        if unique_names:
            index[path.name] = path

    chunk: list[Path] = []
    argument_size = 0

    def flush(group: list[Path]) -> None:
        if not group:
            return
        command = [
            exiftool,
            "-json",
            "-m",
            "-charset",
            "filename=UTF8",
            "-DateTimeOriginal",
            "-SubSecTimeOriginal",
            "-CreateDate",
            "-Model",
            *[str(path) for path in group],
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            timeout=180,
        )
        if completed.returncode not in (0, 1):
            log.warning(
                "exiftool exited %s: %s",
                completed.returncode,
                completed.stderr.decode("utf-8", errors="replace")[:400],
            )
        text = completed.stdout.decode("utf-8", errors="replace").strip()
        if not text:
            return
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = [payload]
        for item in payload:
            if not isinstance(item, dict):
                continue
            source = str(item.get("SourceFile", ""))
            path = index.get(source) or index.get(str(Path(source)))
            if path is None:
                try:
                    path = index.get(str(Path(source).resolve()))
                except OSError:
                    path = None
            if path is None:
                continue
            moment = parse_exif_datetime(
                item.get("DateTimeOriginal") or item.get("CreateDate"),
                item.get("SubSecTimeOriginal"),
            )
            camera = item.get("Model")
            camera_text = str(camera).strip() if camera else None
            found[path] = CaptureInfo(
                moment,
                SOURCE_EXIFTOOL if moment is not None else SOURCE_MISSING,
                camera_text or None,
            )

    for path in paths:
        chunk.append(path)
        argument_size += len(str(path)) + 1
        if argument_size > 60_000 or len(chunk) >= 80:
            flush(chunk)
            chunk = []
            argument_size = 0
    flush(chunk)
    return found


def _read_with_exifread(path: Path) -> CaptureInfo | None:
    try:
        import exifread
    except ImportError:
        return None
    exif_log = logging.getLogger("exifread")
    previous = exif_log.level
    exif_log.setLevel(logging.ERROR)
    try:
        with path.open("rb") as handle:
            tags = exifread.process_file(handle, details=False, strict=False, extract_thumbnail=False)
    except OSError as exc:
        log.warning("Could not read %s: %s", path, exc)
        return None
    except Exception as exc:
        log.debug("exifread could not read %s (%s)", path.name, exc)
        return None
    finally:
        exif_log.setLevel(previous)
    if not tags:
        return None
    moment = parse_exif_datetime(
        _tag_text(tags, "EXIF DateTimeOriginal")
        or _tag_text(tags, "EXIF DateTimeDigitized")
        or _tag_text(tags, "Image DateTime"),
        _tag_text(tags, "EXIF SubSecTimeOriginal")
        or _tag_text(tags, "EXIF SubSecTime")
        or _tag_text(tags, "EXIF SubSecTimeDigitized"),
    )
    camera = _tag_text(tags, "Image Model") or _tag_text(tags, "Image Make")
    if moment is None and camera is None:
        return None
    return CaptureInfo(moment, SOURCE_EXIFREAD if moment is not None else SOURCE_MISSING, camera)


def _tag_text(tags: dict[str, object], key: str) -> str | None:
    value = tags.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _suffix(value: str) -> str:
    text = str(value).strip().lower()
    if not text:
        return ""
    return text if text.startswith(".") else f".{text}"


def _read_with_builtin(path: Path) -> CaptureInfo:
    try:
        with path.open("rb") as handle:
            signature = handle.read(16)
            handle.seek(0)
            if signature.startswith(b"\xff\xd8"):
                data = handle.read(min(2_000_000, path.stat().st_size))
                info = _parse_jpeg(data)
                if info and info.captured_at:
                    return info
                moment = _regex_datetime(data[:1_000_000])
                if moment:
                    return CaptureInfo(moment, SOURCE_HEADER, None)
                return CaptureInfo(None, SOURCE_MISSING, None)

            for limit in (512 * 1024, 8 * 1024 * 1024):
                handle.seek(0)
                data = handle.read(limit)
                info = _info_from_bytes(data)
                if info and info.captured_at:
                    return info
            handle.seek(0)
            moment = _regex_datetime(handle.read(4 * 1024 * 1024))
            if moment:
                return CaptureInfo(moment, SOURCE_HEADER, None)
    except OSError as exc:
        log.warning("Could not read %s: %s", path, exc)
    return CaptureInfo(None, SOURCE_MISSING, None)


def _info_from_bytes(data: bytes) -> CaptureInfo | None:
    info = _from_tiff_bytes(data)
    if info and info.captured_at:
        return info
    embedded = _find_embedded_tiff(data)
    if embedded is not None:
        info = _from_tiff_bytes(embedded)
        if info and info.captured_at:
            return info
    jpeg_info = _parse_jpeg(data) if data.startswith(b"\xff\xd8") else None
    if jpeg_info and jpeg_info.captured_at:
        return jpeg_info
    return None


def _find_embedded_tiff(data: bytes) -> bytes | None:
    index = data.find(_EXIF_MARK)
    if index < 0:
        return None
    return data[index + len(_EXIF_MARK) :]


def _from_tiff_bytes(data: bytes) -> CaptureInfo | None:
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        return None
    endian = "<" if data[:2] == b"II" else ">"
    try:
        magic = struct.unpack_from(endian + "H", data, 2)[0]
        offset = struct.unpack_from(endian + "I", data, 4)[0]
    except struct.error:
        return None
    if magic not in _TIFF_MAGICS:
        return None
    tags = _collect_tags(data, endian, offset, set(), 0)
    raw_dt = tags.get(_DATETIME_ORIGINAL) or tags.get(_DATETIME) or tags.get(_DATETIME_DIGITIZED)
    subsec = tags.get(_SUBSEC_ORIGINAL) or tags.get(_SUBSEC) or tags.get(_SUBSEC_DIGITIZED)
    moment = parse_exif_datetime(raw_dt, subsec)
    model = _clean(tags.get(_MODEL))
    make = _clean(tags.get(_MAKE))
    camera = model or make
    if moment is None and camera is None:
        return None
    return CaptureInfo(moment, SOURCE_EXIF if moment else SOURCE_MISSING, camera)


def _collect_tags(
    data: bytes,
    endian: str,
    offset: int,
    seen: set[int],
    depth: int,
) -> dict[int, str]:
    tags: dict[int, str] = {}
    if depth > 6 or offset in seen or offset < 8 or offset + 2 > len(data):
        return tags
    seen.add(offset)
    try:
        count = struct.unpack_from(endian + "H", data, offset)[0]
    except struct.error:
        return tags
    if count > 400:
        return tags
    base = offset + 2
    for index in range(count):
        pos = base + index * 12
        if pos + 12 > len(data):
            break
        tag, typ, number = struct.unpack_from(endian + "HHI", data, pos)
        if tag == _EXIF_IFD:
            pointer = _pointer(data, endian, typ, pos + 8)
            if pointer:
                nested = _collect_tags(data, endian, pointer, seen, depth + 1)
                for key, value in nested.items():
                    tags.setdefault(key, value)
            continue
        if tag in _INTERESTING_ASCII and typ == 2:
            text = _ascii_value(data, endian, number, pos + 8)
            if text:
                tags.setdefault(tag, text)
    next_pos = base + count * 12
    if next_pos + 4 <= len(data):
        try:
            nxt = struct.unpack_from(endian + "I", data, next_pos)[0]
        except struct.error:
            nxt = 0
        if nxt:
            nested = _collect_tags(data, endian, nxt, seen, depth + 1)
            for key, value in nested.items():
                tags.setdefault(key, value)
    return tags


def _pointer(data: bytes, endian: str, typ: int, field: int) -> int | None:
    try:
        if typ == 4 and field + 4 <= len(data):
            return struct.unpack_from(endian + "I", data, field)[0]
        if typ == 3 and field + 2 <= len(data):
            return struct.unpack_from(endian + "H", data, field)[0]
    except struct.error:
        return None
    return None


def _ascii_value(data: bytes, endian: str, count: int, field: int) -> str | None:
    if count <= 0:
        return None
    try:
        if count <= 4:
            if field + count > len(data):
                return None
            raw = data[field : field + count]
        else:
            if field + 4 > len(data):
                return None
            ptr = struct.unpack_from(endian + "I", data, field)[0]
            if ptr < 0 or ptr + count > len(data):
                return None
            raw = data[ptr : ptr + count]
    except struct.error:
        return None
    text = raw.split(b"\x00", 1)[0].decode("latin1", errors="replace").strip()
    return text or None


def _parse_jpeg(data: bytes) -> CaptureInfo | None:
    if len(data) < 4 or not data.startswith(b"\xff\xd8"):
        return None
    index = 2
    while index + 4 <= len(data):
        if data[index] != 0xFF:
            return None
        index += 1
        while index < len(data) and data[index] == 0xFF:
            index += 1
        if index >= len(data):
            return None
        marker = data[index]
        index += 1
        if marker in (0xD8, 0xD9) or marker == 0xDA:
            return None
        if marker == 0x01 or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(data):
            return None
        segment_length = struct.unpack_from(">H", data, index)[0]
        if segment_length < 2 or index + segment_length > len(data):
            return None
        if marker == 0xE1:
            segment = data[index + 2 : index + segment_length]
            if segment.startswith(_EXIF_MARK):
                info = _from_tiff_bytes(segment[len(_EXIF_MARK) :])
                if info and info.captured_at:
                    return info
        index += segment_length
    return None


def _regex_datetime(data: bytes) -> datetime | None:
    match = _DATE_RE.search(data)
    if match is None:
        return None
    return parse_exif_datetime(match.group().decode("ascii"))


def _apply_subsec(moment: datetime, subsec: str | None) -> datetime:
    if subsec is None:
        return moment
    digits = "".join(character for character in subsec.strip() if character.isdigit())
    if not digits:
        return moment
    digits = digits[:6]
    microseconds = int(digits.ljust(6, "0"))
    return moment.replace(microsecond=microseconds)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip(" \x00")
    return text or None


def _pack_tiff(
    when: datetime,
    subsec: str | None,
    model: str | None,
    *,
    endian: str,
    magic: int,
) -> bytes:
    if endian not in ("<", ">"):
        raise ValueError("endian must be '<' or '>'")
    order = b"II" if endian == "<" else b"MM"
    date_bytes = (when.strftime("%Y:%m:%d %H:%M:%S") + "\x00").encode("ascii")
    model_bytes = ((model or "") + "\x00").encode("ascii") if model else None
    sub_bytes = (str(subsec) + "\x00").encode("ascii") if subsec else None

    ifd0_count = 2 if model_bytes else 1
    exif_count = 2 if sub_bytes else 1
    ifd0_start = 8
    ifd0_size = 2 + ifd0_count * 12 + 4
    exif_start = ifd0_start + ifd0_size
    exif_size = 2 + exif_count * 12 + 4
    strings = exif_start + exif_size
    model_off = strings
    cursor = strings
    if model_bytes and len(model_bytes) > 4:
        cursor += len(model_bytes)
    date_off = cursor
    cursor += len(date_bytes) if len(date_bytes) > 4 else 0
    sub_off = cursor
    total = cursor + (len(sub_bytes) if sub_bytes and len(sub_bytes) > 4 else 0)
    out = bytearray(total)
    out[0:2] = order
    struct.pack_into(endian + "H", out, 2, magic)
    struct.pack_into(endian + "I", out, 4, ifd0_start)
    struct.pack_into(endian + "H", out, ifd0_start, ifd0_count)
    entry = ifd0_start + 2
    if model_bytes:
        _write_ascii(out, entry, _MODEL, model_bytes, model_off, endian)
        entry += 12
    struct.pack_into(endian + "HHI", out, entry, _EXIF_IFD, 4, 1)
    struct.pack_into(endian + "I", out, entry + 8, exif_start)
    struct.pack_into(endian + "I", out, ifd0_start + 2 + ifd0_count * 12, 0)

    struct.pack_into(endian + "H", out, exif_start, exif_count)
    exif_entry = exif_start + 2
    _write_ascii(out, exif_entry, _DATETIME_ORIGINAL, date_bytes, date_off, endian)
    if sub_bytes:
        _write_ascii(out, exif_entry + 12, _SUBSEC_ORIGINAL, sub_bytes, sub_off, endian)
    struct.pack_into(endian + "I", out, exif_start + 2 + exif_count * 12, 0)

    if model_bytes and len(model_bytes) > 4:
        out[model_off : model_off + len(model_bytes)] = model_bytes
    if len(date_bytes) > 4:
        out[date_off : date_off + len(date_bytes)] = date_bytes
    if sub_bytes and len(sub_bytes) > 4:
        out[sub_off : sub_off + len(sub_bytes)] = sub_bytes
    return bytes(out)


def _write_ascii(
    out: bytearray,
    pos: int,
    tag: int,
    data: bytes,
    offset: int,
    endian: str,
) -> None:
    struct.pack_into(endian + "HHI", out, pos, tag, 2, len(data))
    if len(data) <= 4:
        out[pos + 8 : pos + 8 + len(data)] = data
    else:
        struct.pack_into(endian + "I", out, pos + 8, offset)
