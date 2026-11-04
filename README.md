# Outburst

Outburst is a free desktop app for culling wildlife and nature bursts on your own computer. It groups frames that were shot a fraction of a second apart, scores each one for sharpness and exposure, and shows you the frame it would keep.

Nothing is uploaded. There is no account and no subscription. Outburst never modifies your original RAW or JPEG files. A subject model is downloaded only when you turn that feature on and the weights are not already in the models folder.

![Outburst review window with a burst selected and the recommended frame highlighted](docs/images/outburst-review.png)

## What you can do

- Open a folder of RAW bursts, or JPEGs if that is what the camera wrote.
- See the bursts in a list, and the frames in a filmstrip.
- Compare scores and open a larger preview of the recommended frame. It is marked **RECOMMENDED**, outlined in green, and labeled **BEST** on the filmstrip.
- Mark keepers and rejects from the keyboard. Those marks are remembered on this computer.
- Export the keepers as copies or hard links, write XMP sidecars, or save a JSON and CSV report.

The copy above is a small drawn sample so you can try the window without a memory card. Real feathers score higher than these simple shapes. The ranking idea is the same: the sharp frame beats the soft ones and the blown one.

## Install

You need Python 3.11 or newer. Outburst runs on macOS (including Apple Silicon), Windows, and Linux.

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Then start the app:

```bash
outburst
```

Or open a folder immediately:

```bash
outburst examples/sample_burst
```

`examples/sample_burst` is included. To draw it again:

```bash
python examples/generate_sample_burst.py
```

### exiftool

Install [exiftool](https://exiftool.org/) if you can. Outburst uses it first, because it understands the most camera formats, including CR3.

- macOS: `brew install exiftool`
- Windows: the ExifTool Windows package
- Linux: the `libimage-exiftool-perl` package, or the equivalent for your distribution

Without exiftool, Outburst reads capture times with the exifread library that ships with the app. exifread covers JPEG and TIFF-based RAW such as CR2, NEF, ARW, DNG, ORF, PEF, and RW2. It does not read CR3, RAF, or X3F. Those files fall through to a search for a date string, which is less reliable. A frame with no capture time stays in a burst by itself. When a RAW folder is opened without exiftool, the window says which of those formats are in the folder.

### Linux libraries

A normal desktop already has what PySide6 needs. If the window fails to open because `libEGL.so.1` is missing, install `libegl1` (Debian and Ubuntu) and try again.

## The window

**Open** asks for a folder. Outburst looks in that folder only, unless you check **Include subfolders on the next scan**. It reads these RAW types, plus JPEG:

`CR2` `CR3` `NEF` `NRW` `ARW` `ORF` `RAF` `RW2` `DNG` `PEF` `SRW` and the other common raw extensions listed in `src/outburst/scanner.py`.

A JPEG that sits next to a RAW file with the same name is used as the preview. It is not counted as a second frame.

While the scan runs, the status bar shows progress. The first visit reads each file. The next visit is much faster, because measurements and thumbnails are cached on this computer.

The left column lists bursts. The large frame is the one you are reviewing. The strip along the bottom is every frame in the current burst, with a score from 0 to 100.

**Burst gap** is how close two frames must be to stay in one burst. The default is 1.0 second. A gap of exactly 1.0 second starts a new burst. Moving the control regroups immediately.

**Sharpness** and **Center weight** change the ranking immediately, without reading the files again. Sharpness is three quarters of the score by default. The center of the frame counts a bit more than the corners, which suits a bird that is near the middle.

**Subject-aware sharpness** is off unless you turn it on. See below.

### Keyboard

These work unless a slider or the burst-gap box is focused. Arrow keys then adjust that control.

| Keys | Action |
| --- | --- |
| Right / Left | Next or previous frame |
| Down / Up | Next or previous burst |
| Home / End | First or last frame in the burst |
| Enter | Jump to the recommended frame |
| A | Mark this frame as a keeper. Press again to clear it. |
| X | Reject this frame. Press again to clear it. |
| R | Reject the whole burst. Press again to clear it. |
| Ctrl+O | Open a folder |
| F5 | Rescan |
| Ctrl+E | Export keepers |
| Ctrl+Shift+E | Export a JSON and CSV report |
| Ctrl+Shift+X | Write XMP sidecars |
| Esc | Cancel the scan |

## How a frame gets its score

Outburst does not demosaic the RAW file. It scores the embedded JPEG preview (or the paired JPEG), scaled so the long side is at most 1024 pixels.

Sharpness is the variance of a Laplacian on that preview. Higher means more fine detail. A second edge measurement, Tenengrad, is shown as edge energy and is not used for the rank.

Exposure looks at the preview histogram. Large areas of clipped highlights are penalized more than crushed shadows. A little specular highlight is tolerated.

The number under each thumbnail is the combined score, 0 to 100, with sharpness weighted more heavily than exposure. Within a burst, the highest score is the recommendation. If two scores match, the sharper full frame wins, then the earlier filename.

## Your files

Scanning only writes into Outburst's cache:

| System | Cache |
| --- | --- |
| macOS | `~/Library/Caches/Outburst` |
| Windows | `%LOCALAPPDATA%\Outburst\Cache` |
| Linux | `~/.cache/outburst` |

Delete that folder if you want Outburst to measure everything again. A log file, `outburst.log`, is kept there too.

**Keepers** copies the chosen frames into a folder you pick. A frame you marked as a keeper is exported. If you mark several keepers in one burst, each of them is exported. Otherwise Outburst exports the recommendation, unless you rejected it, in which case it takes the next best frame that is still in play. A burst you fully rejected is skipped. Hard links are offered when you want a second name for the same file without using more disk space. If a link cannot be created, Outburst copies the file instead. Deleting the file in the keepers folder leaves the original in place.

**XMP** writes a sidecar named like the photo, with `.xmp` instead of the photo extension. Lightroom-style rating is 5 for a keeper, -1 for a reject, and 0 otherwise. The sidecar also stores `outburst:Score`, the burst id, and whether the frame was the recommendation. Outburst asks before writing. If a sidecar is already there and is real XMP, Outburst updates its own fields and leaves the rest. If the file is not XMP, Outburst does not overwrite it.

**Report** writes `outburst-report.json` and `outburst-report.csv`. `examples/sample_output` shows the shape of both, plus a sample sidecar. Reports you export use full file paths.

## Subject detection

This part is optional and stays off until you turn it on. The scoring panel has a model dropdown: YOLO11 Nano (the default), YOLO11 Small, YOLOv8 Nano, and YOLOv8 Small. These are the official Ultralytics weights.

```bash
pip install "outburst[subject]"
```

Check **Subject-aware sharpness**. If the selected file is not already in the models folder, Outburst downloads it there and shows progress in the status bar. Cancel with Esc. When a bird or animal is found, sharpness is measured inside that box (with a small margin) and the box is drawn on the preview. If nothing is found, or the model cannot be loaded, Outburst keeps the normal full-frame score and tells you.

When you run Outburst from this project, the folder is `models` next to this README. Otherwise the weights go in a `models` folder inside the Outburst cache. Set `OUTBURST_MODELS` to use a different folder. A model marked **ready** is already on disk. A model marked **download** is fetched the next time subject scoring needs it. Photos are still never uploaded.

On Apple Silicon the detector uses MPS when PyTorch provides it. On NVIDIA it uses CUDA. Otherwise it uses the CPU. The rest of Outburst does not need a GPU.

## Apple Silicon

Install Outburst with an arm64 build of Python 3.11 or newer and you do not need Rosetta. PySide6 and rawpy publish macOS wheels for Apple Silicon. The interface is the same app on Intel and Apple Silicon. High-DPI displays are handled by Qt 6.

## For developers

```bash
pip install -e ".[dev]"
pytest
```

Core tests cover grouping, scoring, timestamp reading, preview fallback, the cache, and the non-destructive outputs. A lighter test opens the window offscreen, loads a folder, and moves between frames. The pipeline does not import Qt.

See [docs/architecture.md](docs/architecture.md) for the scoring formula, cache keys, and which modules are allowed to write files.

Packaging as a double-clickable app (pyinstaller or briefcase) is not part of this version. Installing into a virtual environment is the supported way to run Outburst today.

## Roadmap

This version includes the review window, timestamp grouping, preview scoring, a disk cache, adjustable weights, keyboard culling, XMP, keeper export, and reports. Subject detection is optional. The model dropdown downloads a missing weights file into the models folder.

Later versions can add a bundled small detector, a side-by-side compare view, and signed installers for macOS and Windows.

## License

[MIT](LICENSE).
