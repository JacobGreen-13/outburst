"""Non-destructive outputs: XMP sidecars, keeper copies, and reports.

Outburst never opens a photograph for writing. Sidecars are separate files.
Keeper export copies or hard-links the chosen frames into a folder you pick.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from outburst import __version__
from outburst.models import Burst, Frame, FrameAnalysis, Library
from outburst.scoring import recommended_offset

_XMP_NS = "http://ns.adobe.com/xap/1.0/"
_OUTBURST_NS = "https://outburst.app/ns/1.0/"


class SidecarError(RuntimeError):
    """An existing sidecar could not be updated safely."""


@dataclass(frozen=True)
class KeeperChoice:
    burst_id: str
    path: Path
    reason: str


@dataclass(frozen=True)
class ExportResult:
    source: Path
    destination: Path
    method: str
    note: str = ""


def choose_keepers(
    bursts: Sequence[Burst],
    by_path: Mapping[Path, FrameAnalysis],
    decisions: Mapping[Path, str],
    *,
    minimum_frames: int = 2,
) -> list[KeeperChoice]:
    """Return the frames marked as keepers in bursts that meet ``minimum_frames``.

    A shorter burst is hidden in the review list and is not saved. Within a
    burst, marked frames are ordered with the strongest score first.
    """

    minimum = max(1, int(minimum_frames))
    chosen: list[KeeperChoice] = []
    for burst in bursts:
        if burst.count < minimum:
            continue
        marked = [frame for frame in burst.frames if decisions.get(frame.path) == "keep"]
        for frame in sorted(marked, key=lambda item: _rank(item, by_path)):
            chosen.append(KeeperChoice(burst.id, frame.path, "marked"))
    return chosen


def export_keepers(
    sources: Sequence[Path],
    dest_dir: Path,
    mode: str = "copy",
) -> list[ExportResult]:
    """Copy or hard-link keeper files. Existing different files are not overwritten."""

    if mode not in {"copy", "hardlink"}:
        raise ValueError("mode must be 'copy' or 'hardlink'")
    dest_dir.mkdir(parents=True, exist_ok=True)
    reserved: set[str] = set()
    results: list[ExportResult] = []
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination, already = _destination(source, dest_dir, reserved)
        if already:
            results.append(ExportResult(source, destination, "existing", "Already in the keepers folder"))
            continue
        if mode == "hardlink":
            try:
                os.link(source, destination)
            except OSError as exc:
                shutil.copy2(source, destination)
                results.append(
                    ExportResult(
                        source,
                        destination,
                        "copy",
                        f"Hard link unavailable ({exc.__class__.__name__}); copied instead",
                    )
                )
            else:
                results.append(ExportResult(source, destination, "hardlink"))
        else:
            shutil.copy2(source, destination)
            results.append(ExportResult(source, destination, "copy"))
    return results


def write_library_sidecars(
    library: Library,
    decisions: Mapping[Path, str],
    *,
    minimum_frames: int = 2,
) -> tuple[list[Path], list[str]]:
    """Write one XMP sidecar per frame. Returns written paths and warning strings."""

    by_path = {item.frame.path: item for item in library.analyses}
    keepers = {
        choice.path
        for choice in choose_keepers(
            library.bursts,
            by_path,
            decisions,
            minimum_frames=minimum_frames,
        )
    }
    recommended: set[Path] = set()
    burst_of: dict[Path, str] = {}
    for burst in library.bursts:
        if not burst.frames:
            continue
        for frame in burst.frames:
            burst_of[frame.path] = burst.id
        index = recommended_offset(burst.frames, by_path)
        recommended.add(burst.frames[index].path)

    written: list[Path] = []
    warnings: list[str] = []
    for analysis in library.analyses:
        frame = analysis.frame
        decision = decisions.get(frame.path)
        if decision == "reject":
            rating = -1
        elif frame.path in keepers:
            rating = 5
        else:
            rating = 0
        fields = {
            "xmp:Rating": str(rating),
            "outburst:Score": "" if analysis.composite is None else f"{analysis.composite:.4f}",
            "outburst:ScorePercent": ""
            if analysis.composite is None
            else str(int(round(analysis.composite * 100))),
            "outburst:Sharpness": ""
            if analysis.metrics is None
            else f"{analysis.metrics.sharpness:.2f}",
            "outburst:BurstId": burst_of.get(frame.path, ""),
            "outburst:Recommended": "True" if frame.path in recommended else "False",
            "outburst:Decision": decision or "unset",
            "outburst:App": "Outburst",
            "outburst:Version": __version__,
        }
        try:
            written.append(write_xmp_sidecar(frame.path, fields))
        except (OSError, SidecarError) as exc:
            warnings.append(f"{frame.path.name}: {exc}")
    return written, warnings


def write_xmp_sidecar(image_path: Path, fields: Mapping[str, str]) -> Path:
    """Create or update `<stem>.xmp` beside the photograph."""

    sidecar = image_path.with_suffix(".xmp")
    if sidecar.exists():
        text = sidecar.read_text(encoding="utf-8")
        text = upsert_xmp(text, fields)
    else:
        text = fresh_xmp(fields)
    temporary = sidecar.with_name(sidecar.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(sidecar)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return sidecar


def fresh_xmp(fields: Mapping[str, str]) -> str:
    attributes = "\n".join(f'      {key}="{_xml_attr(value)}"' for key, value in fields.items())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '    <rdf:Description rdf:about=""\n'
        f'      xmlns:xmp="{_XMP_NS}"\n'
        f'      xmlns:outburst="{_OUTBURST_NS}"\n'
        f"{attributes}/>\n"
        "  </rdf:RDF>\n"
        "</x:xmpmeta>\n"
    )


def upsert_xmp(text: str, fields: Mapping[str, str]) -> str:
    """Update Outburst fields in an existing sidecar and leave other text in place."""

    if "rdf:Description" not in text or "xmpmeta" not in text:
        raise SidecarError("existing sidecar is not XMP, so Outburst left it alone")
    updated = text
    updated = _ensure_xmlns(updated, "xmlns:xmp", _XMP_NS)
    updated = _ensure_xmlns(updated, "xmlns:outburst", _OUTBURST_NS)
    for key, value in fields.items():
        updated = _upsert_attr(updated, key, value)
    return updated


def write_reports(
    library: Library,
    decisions: Mapping[Path, str],
    dest_dir: Path,
    *,
    minimum_frames: int = 2,
) -> tuple[Path, Path]:
    """Write `outburst-report.json` and `outburst-report.csv`."""

    dest_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(library, decisions, minimum_frames=minimum_frames)
    json_path = dest_dir / "outburst-report.json"
    csv_path = dest_dir / "outburst-report.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows = report["frames"]
    fieldnames = list(rows[0].keys()) if rows else list(_empty_row().keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def build_report(
    library: Library,
    decisions: Mapping[Path, str],
    *,
    minimum_frames: int = 2,
) -> dict[str, object]:
    by_path = {item.frame.path: item for item in library.analyses}
    keepers = choose_keepers(library.bursts, by_path, decisions, minimum_frames=minimum_frames)
    keeper_paths = {choice.path for choice in keepers}
    frames: list[dict[str, object]] = []
    bursts_out: list[dict[str, object]] = []
    for burst in library.bursts:
        recommended_path = None
        if burst.frames:
            recommended_path = burst.frames[recommended_offset(burst.frames, by_path)].path
        bursts_out.append(
            {
                "id": burst.id,
                "count": burst.count,
                "started_at": _iso(burst.started_at),
                "recommended": None if recommended_path is None else str(recommended_path),
            }
        )
        for index, frame in enumerate(burst.frames, start=1):
            frames.append(_frame_row(burst.id, index, frame, by_path, decisions, recommended_path, keeper_paths))
    return {
        "app": "Outburst",
        "version": __version__,
        "folder": str(library.folder),
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "threshold_seconds": library.threshold,
        "weights": {
            "sharpness": library.weights.sharpness,
            "exposure": library.weights.exposure,
            "center_mix": library.weights.center_mix,
            "prefer_subject": library.weights.prefer_subject,
        },
        "bursts": bursts_out,
        "keepers": [str(choice.path) for choice in keepers],
        "frames": frames,
    }


def _frame_row(
    burst_id: str,
    frame_index: int,
    frame: Frame,
    by_path: Mapping[Path, FrameAnalysis],
    decisions: Mapping[Path, str],
    recommended_path: Path | None,
    keeper_paths: set[Path],
) -> dict[str, object]:
    analysis = by_path.get(frame.path)
    metrics = None if analysis is None else analysis.metrics
    row = _empty_row()
    row.update(
        {
            "burst_id": burst_id,
            "frame_index": frame_index,
            "filename": frame.path.name,
            "path": str(frame.path),
            "captured_at": _iso(frame.captured_at),
            "camera": frame.camera or "",
            "score": None if analysis is None or analysis.composite is None else round(analysis.composite, 4),
            "sharpness": None if metrics is None else round(metrics.sharpness, 2),
            "center_sharpness": None if metrics is None else round(metrics.center_sharpness, 2),
            "tenengrad": None if metrics is None else round(metrics.tenengrad, 2),
            "exposure": None if metrics is None else round(metrics.exposure, 4),
            "highlight_clip": None if metrics is None else round(metrics.highlight_clip, 4),
            "shadow_clip": None if metrics is None else round(metrics.shadow_clip, 4),
            "subject_sharpness": None
            if metrics is None or metrics.subject_sharpness is None
            else round(metrics.subject_sharpness, 2),
            "subject_label": "" if metrics is None or not metrics.subject_label else metrics.subject_label,
            "recommended": frame.path == recommended_path,
            "keeper": frame.path in keeper_paths,
            "decision": decisions.get(frame.path, "unset"),
            "error": "" if analysis is None or not analysis.error else analysis.error,
        }
    )
    return row


def _empty_row() -> dict[str, object]:
    return {
        "burst_id": "",
        "frame_index": 0,
        "filename": "",
        "path": "",
        "captured_at": "",
        "camera": "",
        "score": None,
        "sharpness": None,
        "center_sharpness": None,
        "tenengrad": None,
        "exposure": None,
        "highlight_clip": None,
        "shadow_clip": None,
        "subject_sharpness": None,
        "subject_label": "",
        "recommended": False,
        "keeper": False,
        "decision": "unset",
        "error": "",
    }


def _rank(frame: Frame, by_path: Mapping[Path, FrameAnalysis]) -> tuple[float, float, str]:
    analysis = by_path.get(frame.path)
    composite = -1.0 if analysis is None or analysis.composite is None else analysis.composite
    sharpness = -1.0 if analysis is None or analysis.metrics is None else analysis.metrics.sharpness
    return (-composite, -sharpness, frame.path.name.lower())


def _destination(source: Path, dest_dir: Path, reserved: set[str]) -> tuple[Path, bool]:
    stem = source.stem
    suffix = source.suffix
    candidates = [dest_dir / source.name]
    candidates.extend(dest_dir / f"{stem}_{number}{suffix}" for number in range(1, 1000))
    for candidate in candidates:
        if candidate.exists():
            try:
                if candidate.samefile(source):
                    reserved.add(candidate.name.lower())
                    return candidate, True
            except OSError:
                pass
            continue
        if candidate.name.lower() in reserved:
            continue
        reserved.add(candidate.name.lower())
        return candidate, False
    raise OSError(f"Could not find a free filename for {source.name} in {dest_dir}")


def _upsert_attr(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(\s{re.escape(key)}\s*=\s*\")([^\"]*)(\")")
    if pattern.search(text):
        return pattern.sub(lambda match: match.group(1) + _xml_attr(value) + match.group(3), text, count=1)
    return _insert_into_description(text, f'{key}="{_xml_attr(value)}"')


def _ensure_xmlns(text: str, name: str, uri: str) -> str:
    if re.search(rf"\s{re.escape(name)}\s*=", text):
        return text
    return _insert_into_description(text, f'{name}="{uri}"')


def _insert_into_description(text: str, attribute: str) -> str:
    match = re.search(r"<rdf:Description\b", text)
    if match is None:
        raise SidecarError("XMP has no rdf:Description element")
    index = match.end()
    quote: str | None = None
    while index < len(text):
        character = text[index]
        if quote:
            if character == quote:
                quote = None
        elif character in {'"', "'"}:
            quote = character
        elif character == ">":
            return text[:index] + " " + attribute + text[index:]
        index += 1
    raise SidecarError("XMP description tag is incomplete")


def _xml_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _iso(moment: datetime | None) -> str:
    if moment is None:
        return ""
    return moment.isoformat(timespec="microseconds")
