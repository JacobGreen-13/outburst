"""Non-destructive outputs: XMP sidecars, keeper copies, and reports.

Outburst never opens a photograph for writing. Sidecars are separate files.
Keeper export copies or hard-links the chosen frames into a folder you pick.

``choose_keepers`` and ``export_keepers`` save marked frames from bursts that meet the minimum length. ``write_exported_sidecars`` writes one ``.xmp`` beside each saved keeper. ``frames_for_sidecars``, ``review_xmp_fields``, ``sidecar_rating``, and ``write_library_sidecars`` write XMP for those keepers and for any star-rated frame. ``write_xmp_sidecar``, ``fresh_xmp``, and ``upsert_xmp`` create or update one sidecar. ``write_reports`` and ``build_report`` write JSON and CSV.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from outburst import __version__
from outburst.models import Burst, Frame, FrameAnalysis, Library
from outburst.parallel import EXPORT_WORKERS, XMP_WORKERS, run_ordered
from outburst.scoring import recommended_offset

_XMP_NS = "http://ns.adobe.com/xap/1.0/"
_XMPDM_NS = "http://ns.adobe.com/xmp/1.0/DynamicMedia/"
_MICROSOFT_PHOTO_NS = "http://ns.microsoft.com/photo/1.0/"
_DIGIKAM_NS = "http://www.digikam.org/ns/1.0/"
_OUTBURST_NS = "https://outburst.app/ns/1.0/"
_MICROSOFT_STAR_PERCENT = {0: "0", 1: "1", 2: "25", 3: "50", 4: "75", 5: "99"}
_XMP_NAMESPACES = (
    ("xmlns:xmp", _XMP_NS),
    ("xmlns:xmpDM", _XMPDM_NS),
    ("xmlns:MicrosoftPhoto", _MICROSOFT_PHOTO_NS),
    ("xmlns:digiKam", _DIGIKAM_NS),
    ("xmlns:outburst", _OUTBURST_NS),
)


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
    sidecar: Path | None = None
    sidecar_note: str = ""


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

    Inputs: bursts, the grouped frames; by_path, analyses keyed by path; decisions, path to keep or reject; minimum_frames, the shortest burst that can be saved.
    Output: KeeperChoice rows for marked keepers in bursts at least that long, strongest score first.
    Uses: Save keepers, the report, and the XMP writer.
    Limitations: a burst shorter than the minimum is skipped even when a frame is marked keep. Rejects are not included. Original photographs are not opened for writing.
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
    progress: Callable[[int, int, str], None] | None = None,
) -> list[ExportResult]:
    """Copy or hard-link keeper files. Existing different files are not overwritten.

    Filenames are chosen on this thread. The copies themselves run two at a time.

    Inputs: sources, the keeper paths to export; dest_dir, the folder the photographer picked; mode, `copy` or `hardlink`; progress, an optional callback of count, total, and a status line.
    Output: one ExportResult per source, in source order.
    Uses: Save keepers.
    Limitations: originals are never opened for writing. A hard link that fails is copied instead. A destination that is already the same file is left in place, and a different existing name gets a numbered name. Two workers write at a time.
    """

    if mode not in {"copy", "hardlink"}:
        raise ValueError("mode must be 'copy' or 'hardlink'")
    dest_dir.mkdir(parents=True, exist_ok=True)
    reserved: set[str] = set()
    planned: list[tuple[Path, Path, bool]] = []
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination, already = _destination(source, dest_dir, reserved)
        planned.append((source, destination, already))

    def copy_at(index: int) -> ExportResult:
        """Copy or hard-link one planned keeper into the destination folder.

        Inputs: index, the position in the planned list of source, destination, and whether that destination is already the same file.
        Output: an ExportResult whose method is existing, hardlink, or copy.
        Uses: export_keepers. The pool calls this on a worker and delivers results in order.
        Limitations: the original is not opened for writing. A failed hard link falls back to a copy. An existing same file is not copied again.
        """

        source, destination, already = planned[index]
        if already:
            return ExportResult(source, destination, "existing", "Already in the keepers folder")
        if mode == "hardlink":
            try:
                os.link(source, destination)
            except OSError as exc:
                shutil.copy2(source, destination)
                return ExportResult(
                    source,
                    destination,
                    "copy",
                    f"Hard link unavailable ({exc.__class__.__name__}); copied instead",
                )
            return ExportResult(source, destination, "hardlink")
        shutil.copy2(source, destination)
        return ExportResult(source, destination, "copy")

    def report(index: int, _result: ExportResult) -> None:
        """Forward one finished export to the progress callback.

        Inputs: index, the finished source; _result, the ExportResult, which is unused beyond ordering.
        Output: none. When progress is set, it is called with the 1-based count, the total, and the source filename.
        Uses: export_keepers, on the coordinating thread after each copy finishes in order.
        Limitations: a missing progress callback does nothing. This does not write a file.
        """

        if progress is not None:
            progress(index + 1, len(planned), f"Saving {planned[index][0].name}")

    return run_ordered(len(planned), copy_at, workers=EXPORT_WORKERS, on_finished=report)


def _library_marks(library: Library) -> tuple[dict[Path, FrameAnalysis], dict[Path, str], set[Path]]:
    """Index a library by path, burst, and recommended frame.

    Inputs: library, the scanned folder.
    Output: analyses keyed by path, burst id keyed by path, and the recommended paths.
    Uses: write_library_sidecars and write_exported_sidecars.
    Limitations: an empty burst adds no paths. This does not write a file.
    """

    by_path = {item.frame.path: item for item in library.analyses}
    burst_of: dict[Path, str] = {}
    recommended: set[Path] = set()
    for burst in library.bursts:
        if not burst.frames:
            continue
        for frame in burst.frames:
            burst_of[frame.path] = burst.id
        recommended.add(burst.frames[recommended_offset(burst.frames, by_path)].path)
    return by_path, burst_of, recommended


def _xmp_fields(
    analysis: FrameAnalysis | None,
    *,
    decision: str | None,
    stars: int | None,
    burst_id: str,
    recommended: bool,
    pick: str | None,
) -> dict[str, str | None]:
    """Build the XMP attributes for one frame.

    Inputs: analysis, the frame measurement or None; decision, the stored keep or reject; stars, a rating or None; burst_id; recommended, whether this frame is the burst recommendation; pick, the flag written as a pick, which may differ from decision for a short-burst keep.
    Output: the review fields plus the Outburst score, burst, and version attributes.
    Uses: write_library_sidecars and write_exported_sidecars.
    Limitations: this returns a dict. It does not write a file. A missing analysis leaves the score fields empty.
    """

    composite = None if analysis is None else analysis.composite
    sharpness = None if analysis is None or analysis.metrics is None else analysis.metrics.sharpness
    return {
        **review_xmp_fields(pick, stars),
        "outburst:Score": "" if composite is None else f"{composite:.4f}",
        "outburst:ScorePercent": "" if composite is None else str(int(round(composite * 100))),
        "outburst:Sharpness": "" if sharpness is None else f"{sharpness:.2f}",
        "outburst:BurstId": burst_id,
        "outburst:Recommended": "True" if recommended else "False",
        "outburst:Decision": decision or "unset",
        "outburst:App": "Outburst",
        "outburst:Version": __version__,
    }


def write_exported_sidecars(
    library: Library,
    results: Sequence[ExportResult],
    decisions: Mapping[Path, str],
    *,
    stars: Mapping[Path, int] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[ExportResult], list[str]]:
    """Write one ``.xmp`` beside each saved keeper.

    The sidecar uses the destination filename. A numbered copy gets a matching sidecar. Nothing is written beside the original.

    Inputs: library, the scanned folder; results, the saved keepers; decisions, path to keep or reject; stars, optional ratings; progress, an optional callback of count, total, and a status line.
    Output: the same results with the sidecar path filled in, and warning strings for sidecars that could not be written.
    Uses: Save keepers, after the photographs have been copied or linked.
    Limitations: originals are never opened for writing. A destination that is not XMP is left alone and reported. The photograph copy is kept when the sidecar fails. Two workers write the sidecars.
    """

    by_path, burst_of, recommended = _library_marks(library)
    star_ratings = {} if stars is None else stars
    updated: list[ExportResult] = []
    warnings: list[str] = []

    def write_at(index: int) -> tuple[ExportResult, str | None]:
        """Write the sidecar for one saved keeper.

        Inputs: index, the position in the export results.
        Output: the result with its sidecar path, or a note when the write raises OSError or SidecarError.
        Uses: write_exported_sidecars. The pool calls this on a worker.
        Limitations: the photograph is not opened for writing. A non-XMP sidecar is left unchanged. Other exceptions propagate.
        """

        result = results[index]
        fields = _xmp_fields(
            by_path.get(result.source),
            decision=decisions.get(result.source, "keep"),
            stars=star_ratings.get(result.source),
            burst_id=burst_of.get(result.source, ""),
            recommended=result.source in recommended,
            pick="keep",
        )
        try:
            sidecar = write_xmp_sidecar(result.destination, fields)
        except (OSError, SidecarError) as exc:
            note = f"{result.destination.with_suffix('.xmp').name}: {exc}"
            return replace(result, sidecar_note=note), note
        return replace(result, sidecar=sidecar), None

    def collect(index: int, item: tuple[ExportResult, str | None]) -> None:
        """Record one finished export sidecar and report progress.

        Inputs: index, the finished result; item, the updated result and an optional warning.
        Output: none. Results and warnings are appended in completion order.
        Uses: write_exported_sidecars.
        Limitations: progress is optional. This does not write the sidecar itself.
        """

        result, warning = item
        updated.append(result)
        if warning is not None:
            warnings.append(warning)
        if progress is not None:
            progress(index + 1, len(results), f"Writing {result.destination.with_suffix('.xmp').name}")

    run_ordered(len(results), write_at, workers=XMP_WORKERS, on_finished=collect)
    return updated, warnings


def sidecar_rating(decision: str | None, stars: int | None) -> int:
    """The ``xmp:Rating`` value for one frame.

    A rating the photographer set is written as that many stars, from 0 to 5.
    With no rating of their own, a pick is 5 stars and a reject is -1 so apps
    that only read ``xmp:Rating`` still see the reject. A reject that also has
    stars keeps those stars here; the reject flag is ``xmpDM:pick``.

    Inputs: decision, `keep`, `reject`, or None; stars, a rating from 0 to 5, or None when the photographer did not set one.
    Output: an integer rating. A reject with no stars, or a reject whose stars are 0, is -1. An unflagged frame with no stars is 0.
    Uses: review_xmp_fields when it fills xmp:Rating.
    Limitations: this returns a number only. It does not write a sidecar. A reject without stars is not given a sidecar by the library writer.
    """

    if stars is not None:
        if decision == "reject" and stars == 0:
            return -1
        return stars
    if decision == "keep":
        return 5
    if decision == "reject":
        return -1
    return 0


def review_xmp_fields(decision: str | None, stars: int | None) -> dict[str, str | None]:
    """Star rating and pick flag in the XMP fields other editors already read.

    ``xmp:Rating`` is the star rating (``-1`` only for a reject that has no stars).
    ``xmpDM:pick`` is 1 for a pick, -1 for a reject, and 0 when the frame is unflagged.
    ``xmpDM:good`` is the Lightroom companion of that flag. ``MicrosoftPhoto:Rating``
    is the same stars as a percent, which Windows and some catalogs read.
    ``digiKam:PickLabel`` is 3 for a pick, 1 for a reject, and 0 otherwise.
    A value of None removes that attribute from an existing sidecar.

    Inputs: decision, `keep`, `reject`, or None; stars, 0 through 5, or None.
    Output: a dict of xmp:Rating, MicrosoftPhoto:Rating, xmpDM:pick, xmpDM:good, and digiKam:PickLabel. xmpDM:good is None when the frame is unflagged.
    Uses: write_library_sidecars.
    Limitations: stars outside 0..5 raise ValueError. This does not write a file. A short-burst keep is not passed here as a pick.
    """

    if stars is not None and stars not in range(6):
        raise ValueError(f"Star rating must be 0 through 5, not {stars}")
    rating = sidecar_rating(decision, stars)
    star_count = stars if stars is not None else (5 if decision == "keep" else 0)
    if decision == "keep":
        pick, good, digikam = "1", "true", "3"
    elif decision == "reject":
        pick, good, digikam = "-1", "false", "1"
    else:
        pick, good, digikam = "0", None, "0"
    return {
        "xmp:Rating": str(rating),
        "MicrosoftPhoto:Rating": _MICROSOFT_STAR_PERCENT[star_count],
        "xmpDM:pick": pick,
        "xmpDM:good": good,
        "digiKam:PickLabel": digikam,
    }


def frames_for_sidecars(
    library: Library,
    decisions: Mapping[Path, str],
    *,
    stars: Mapping[Path, int] | None = None,
    minimum_frames: int = 2,
) -> list[Path]:
    """Keepers that would be saved, plus every frame that has a star rating.

    A star rating is enough on its own. The frame does not also have to be a
    keeper. A reject with no stars is omitted, and so is every other frame.

    Inputs: library, the scanned folder; decisions, path to keep or reject; stars, optional path to a rating; minimum_frames, the shortest burst whose keepers are saved.
    Output: the photograph paths that should receive a sidecar, in analysis order.
    Uses: write_library_sidecars and the count shown before writing XMP.
    Limitations: XMP is only for keepers that would be saved and for frames with a star rating. Rejects without stars are skipped. A short-burst keep is included only when it has stars.
    """

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
    star_ratings = {} if stars is None else stars
    return [
        analysis.frame.path
        for analysis in library.analyses
        if analysis.frame.path in keepers or star_ratings.get(analysis.frame.path) is not None
    ]


def write_library_sidecars(
    library: Library,
    decisions: Mapping[Path, str],
    *,
    stars: Mapping[Path, int] | None = None,
    minimum_frames: int = 2,
    progress: Callable[[int, int, str], None] | None = None,
) -> tuple[list[Path], list[str]]:
    """Write XMP sidecars for keepers and star-rated frames.

    Returns written paths and warning strings. Frames outside that set are
    left alone, including any sidecar already beside them.

    Inputs: library, the scanned folder; decisions, path to keep or reject; stars, optional ratings; minimum_frames, the shortest burst whose keepers are saved; progress, an optional callback of count, total, and a status line.
    Output: the written sidecar paths and the warning strings.
    Uses: the Write XMP action.
    Limitations: originals are never opened for writing. Rejects without stars are skipped. A short-burst keep is not a pick; stars on that frame are still written, with the pick left unset. Two workers write the sidecars. A file that is not XMP is left alone and reported.
    """

    by_path, burst_of, recommended = _library_marks(library)
    keepers = {
        choice.path
        for choice in choose_keepers(
            library.bursts,
            by_path,
            decisions,
            minimum_frames=minimum_frames,
        )
    }
    selected = set(
        frames_for_sidecars(
            library,
            decisions,
            stars=stars,
            minimum_frames=minimum_frames,
        )
    )
    star_ratings = {} if stars is None else stars
    jobs: list[tuple[Path, dict[str, str | None]]] = []
    for analysis in library.analyses:
        frame = analysis.frame
        if frame.path not in selected:
            continue
        decision = decisions.get(frame.path)
        # A keeper in a hidden short burst is not saved, so the sidecar does not flag it as a pick.
        exported = None if frame.path not in keepers and decision == "keep" else decision
        jobs.append(
            (
                frame.path,
                _xmp_fields(
                    analysis,
                    decision=decision,
                    stars=star_ratings.get(frame.path),
                    burst_id=burst_of.get(frame.path, ""),
                    recommended=frame.path in recommended,
                    pick=exported,
                ),
            )
        )

    def write_at(index: int) -> tuple[Path | None, str | None]:
        """Write one sidecar and turn a safe failure into a warning string.

        Inputs: index, the position in the job list of a photograph path and its XMP fields.
        Output: the sidecar path and None, or None and a warning when the write raises OSError or SidecarError.
        Uses: write_library_sidecars. The pool calls this on a worker.
        Limitations: the photograph is not opened for writing. A non-XMP sidecar is left unchanged. Other exceptions propagate.
        """

        path, fields = jobs[index]
        try:
            return write_xmp_sidecar(path, fields), None
        except (OSError, SidecarError) as exc:
            return None, f"{path.name}: {exc}"

    written: list[Path] = []
    warnings: list[str] = []

    def collect(index: int, item: tuple[Path | None, str | None]) -> None:
        """Record one finished sidecar on the coordinating thread and report progress.

        Inputs: index, the finished job; item, the path and optional warning returned by write_at.
        Output: none. Written paths and warnings are appended in completion order.
        Uses: write_library_sidecars.
        Limitations: progress is optional. This does not write the sidecar itself.
        """

        path, warning = item
        if path is not None:
            written.append(path)
        if warning is not None:
            warnings.append(warning)
        if progress is not None:
            progress(index + 1, len(jobs), f"Writing {jobs[index][0].name}")

    run_ordered(len(jobs), write_at, workers=XMP_WORKERS, on_finished=collect)
    return written, warnings


def write_xmp_sidecar(image_path: Path, fields: Mapping[str, str | None]) -> Path:
    """Create or update `<stem>.xmp` beside the photograph.

    Inputs: image_path, the photograph the sidecar sits beside; fields, XMP attributes to set. A None value removes that attribute from an existing sidecar.
    Output: the sidecar path. The text is written to a temporary file and then replaces the sidecar.
    Uses: write_library_sidecars for each selected frame.
    Limitations: the photograph is not opened for writing. An existing file that is not XMP raises SidecarError and is left in place. A failed write deletes the temporary file and reraises.
    """

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


def fresh_xmp(fields: Mapping[str, str | None]) -> str:
    """Build a new XMP document containing only the fields that have a value.

    Inputs: fields, attribute name to value. None values are omitted.
    Output: an XMP string with the Outburst namespaces and those attributes.
    Uses: write_xmp_sidecar when no sidecar exists yet.
    Limitations: this returns text. It does not write a file or open the photograph.
    """

    attributes = "\n".join(
        f'      {key}="{_xml_attr(value)}"' for key, value in fields.items() if value is not None
    )
    namespaces = "\n".join(f'      {name}="{uri}"' for name, uri in _XMP_NAMESPACES)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
        '  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        "    <rdf:Description rdf:about=\"\"\n"
        f"{namespaces}\n"
        f"{attributes}/>\n"
        "  </rdf:RDF>\n"
        "</x:xmpmeta>\n"
    )


def upsert_xmp(text: str, fields: Mapping[str, str | None]) -> str:
    """Update Outburst fields in an existing sidecar and leave other text in place.

    Inputs: text, the current sidecar; fields, attributes to set or, when the value is None, remove.
    Output: the updated XMP text.
    Uses: write_xmp_sidecar when a sidecar is already beside the photograph.
    Limitations: text without rdf:Description or xmpmeta raises SidecarError and is not rewritten. Only the named attributes and the required xmlns declarations change.
    """

    if "rdf:Description" not in text or "xmpmeta" not in text:
        raise SidecarError("existing sidecar is not XMP, so Outburst left it alone")
    updated = text
    for name, uri in _XMP_NAMESPACES:
        updated = _ensure_xmlns(updated, name, uri)
    for key, value in fields.items():
        if value is None:
            updated = _remove_attr(updated, key)
        else:
            updated = _upsert_attr(updated, key, value)
    return updated


def write_reports(
    library: Library,
    decisions: Mapping[Path, str],
    dest_dir: Path,
    *,
    stars: Mapping[Path, int] | None = None,
    minimum_frames: int = 2,
) -> tuple[Path, Path]:
    """Write `outburst-report.json` and `outburst-report.csv`.

    Inputs: library, the scanned folder; decisions, path to keep or reject; dest_dir, the folder the photographer picked; stars, optional ratings; minimum_frames, the shortest burst whose keepers are listed as saved.
    Output: the JSON path and the CSV path.
    Uses: the Save manifest action.
    Limitations: the files are created in dest_dir. Photographs are not modified. Keepers follow the same minimum-length rule as Save keepers.
    """

    dest_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(library, decisions, stars=stars, minimum_frames=minimum_frames)
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
    stars: Mapping[Path, int] | None = None,
    minimum_frames: int = 2,
) -> dict[str, object]:
    """Assemble the JSON-ready report for one library.

    Inputs: library, the scanned folder; decisions, path to keep or reject; stars, optional ratings; minimum_frames, the shortest burst whose marked frames count as keepers.
    Output: a dict with the app, version, folder, weights, bursts, keepers, and one row per frame.
    Uses: write_reports.
    Limitations: paths are stored as text. The recommended frame follows the eye ranking when two or more eyes are rated, otherwise the composite. No photograph is written.
    """

    by_path = {item.frame.path: item for item in library.analyses}
    keepers = choose_keepers(library.bursts, by_path, decisions, minimum_frames=minimum_frames)
    keeper_paths = {choice.path for choice in keepers}
    star_ratings = {} if stars is None else stars
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
            frames.append(
                _frame_row(
                    burst.id,
                    index,
                    frame,
                    by_path,
                    decisions,
                    recommended_path,
                    keeper_paths,
                    star_ratings.get(frame.path),
                )
            )
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
    stars: int | None,
) -> dict[str, object]:
    """Build one report row for a frame.

    Inputs: burst_id and frame_index, the burst identity and 1-based position; frame, the photograph; by_path, analyses; decisions, keep or reject; recommended_path, the burst's recommended frame; keeper_paths, frames that would be saved; stars, the rating or None.
    Output: a dict of the CSV columns, including score, sharpness, subject, decision, stars, and flag.
    Uses: build_report.
    Limitations: a missing analysis leaves the measurements empty. This does not read or write the photograph.
    """

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
            "stars": stars,
            "flag": _flag_name(decisions.get(frame.path)),
            "error": "" if analysis is None or not analysis.error else analysis.error,
        }
    )
    return row


def _empty_row() -> dict[str, object]:
    """Return the report columns with empty defaults.

    Inputs: none.
    Output: a dict whose keys are the CSV field names.
    Uses: _frame_row, and write_reports when a library has no frame rows yet.
    Limitations: the defaults are placeholders. Decision starts as unset and the flag is blank.
    """

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
        "stars": None,
        "flag": "",
        "error": "",
    }


def _flag_name(decision: str | None) -> str:
    """Name the pick or reject flag for a report row.

    Inputs: decision, `keep`, `reject`, or None.
    Output: `pick`, `reject`, or an empty string.
    Uses: _frame_row.
    Limitations: any other decision text is reported as blank. This does not write XMP.
    """

    if decision == "keep":
        return "pick"
    if decision == "reject":
        return "reject"
    return ""


def _rank(frame: Frame, by_path: Mapping[Path, FrameAnalysis]) -> tuple[float, float, str]:
    """Sort key that puts the strongest marked keeper first.

    Inputs: frame, one photograph; by_path, analyses keyed by path.
    Output: a tuple of negated composite, negated full-frame sharpness, and the lowercase filename.
    Uses: choose_keepers.
    Limitations: a missing score sorts last among the marked frames. Eye sharpness is not part of this order.
    """

    analysis = by_path.get(frame.path)
    composite = -1.0 if analysis is None or analysis.composite is None else analysis.composite
    sharpness = -1.0 if analysis is None or analysis.metrics is None else analysis.metrics.sharpness
    return (-composite, -sharpness, frame.path.name.lower())


def _destination(source: Path, dest_dir: Path, reserved: set[str]) -> tuple[Path, bool]:
    """Choose a free filename in the keepers folder.

    Inputs: source, the original file; dest_dir, the destination folder; reserved, lowercase names already claimed in this export.
    Output: the destination path, and True when that path is already the same file as source.
    Uses: export_keepers, on the coordinating thread, before the copies start.
    Limitations: a different existing file is not overwritten. Numbered names run from 1 to 999. The original is not opened for writing. A full set of names raises OSError.
    """

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


def _remove_attr(text: str, key: str) -> str:
    """Delete the first occurrence of one XML attribute.

    Inputs: text, sidecar text; key, the attribute name, including its prefix.
    Output: text with that attribute removed once.
    Uses: upsert_xmp when a field value is None, which drops xmpDM:good on an unflagged frame.
    Limitations: only the first match is removed. The pattern expects double quotes.
    """

    pattern = re.compile(rf"\s{re.escape(key)}\s*=\s*\"[^\"]*\"")
    return pattern.sub("", text, count=1)


def _upsert_attr(text: str, key: str, value: str) -> str:
    """Replace one XML attribute or insert it when it is missing.

    Inputs: text, sidecar text; key, the attribute name; value, the new text, before escaping.
    Output: the updated text.
    Uses: upsert_xmp for ratings, pick flags, and the outburst fields.
    Limitations: only the first existing match is replaced. The value is escaped for an XML attribute.
    """

    pattern = re.compile(rf"(\s{re.escape(key)}\s*=\s*\")([^\"]*)(\")")
    if pattern.search(text):
        return pattern.sub(lambda match: match.group(1) + _xml_attr(value) + match.group(3), text, count=1)
    return _insert_into_description(text, f'{key}="{_xml_attr(value)}"')


def _ensure_xmlns(text: str, name: str, uri: str) -> str:
    """Add a namespace declaration when the description does not already name it.

    Inputs: text, sidecar text; name, the xmlns attribute; uri, the namespace URI.
    Output: the text unchanged when the name is already declared, otherwise with the declaration inserted.
    Uses: upsert_xmp so an older sidecar gains the Outburst and catalog namespaces.
    Limitations: an existing declaration is not rewritten, even if its URI differs. A description that cannot take an attribute raises SidecarError.
    """

    if re.search(rf"\s{re.escape(name)}\s*=", text):
        return text
    return _insert_into_description(text, f'{name}="{uri}"')


def _insert_into_description(text: str, attribute: str) -> str:
    """Insert one attribute into the first rdf:Description start tag.

    Inputs: text, sidecar text; attribute, a name and value fragment that is already escaped.
    Output: the text with the attribute added before the tag closes.
    Uses: the XMP upsert helpers.
    Limitations: a missing or unclosed rdf:Description raises SidecarError. Quoted characters inside the tag are skipped so a closing angle bracket in a value is not treated as the end.
    """

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
            if index > 0 and text[index - 1] == "/":
                return text[: index - 1] + " " + attribute + "/>" + text[index + 1 :]
            return text[:index] + " " + attribute + text[index:]
        index += 1
    raise SidecarError("XMP description tag is incomplete")


def _xml_attr(value: str) -> str:
    """Escape a string for an XML attribute value.

    Inputs: value, the raw text.
    Output: value with ampersand, double quote, and angle brackets replaced by entities.
    Uses: every XMP attribute this module writes.
    Limitations: apostrophes are left as-is because the attributes use double quotes.
    """

    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _iso(moment: datetime | None) -> str:
    """Format a capture time for the report.

    Inputs: moment, a datetime, or None.
    Output: an ISO string with microseconds, or an empty string when moment is None.
    Uses: build_report for burst start times and frame capture times.
    Limitations: no time zone is added when the datetime is naive. A missing clock stays blank.
    """

    if moment is None:
        return ""
    return moment.isoformat(timespec="microseconds")
