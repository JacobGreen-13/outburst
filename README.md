# Outburst

Outburst culls wildlife and nature bursts on your own computer. It groups frames shot a fraction of a second apart, scores each one, and opens on the frame it would keep.

Nothing is uploaded. There is no account. Original RAW and JPEG files stay untouched. A subject or bird-eye model is downloaded only when you turn that feature on and the weights are not already on disk.

![Outburst review window, with the recommended frame marked as a keeper](docs/images/outburst-review.png)

The picture is the included sample, `examples/sample_burst`. Real feathers score higher than these drawn shapes. The sharp frame still beats the soft ones and the blown one.

![Two frames from the same burst, side by side](docs/images/outburst-compare.png)

Press **C** to put the current frame beside another one. It starts beside the best other frame, or the next sharpest when you are already on the best. **[** and **]** move that other frame. Shift-click a thumbnail to compare with it.

## What you can do

- Open a folder of RAW bursts, or JPEGs when that is what the camera wrote.
- Review bursts in the list and frames in the filmstrip. **Minimum** (default 2 frames) hides shorter bursts from the list and from Save. **Minimum score** hides frames under the number you set. A mark on a frame hidden that way stays, and Save still writes it.
- The recommended frame starts as a keeper: a green **KEEPER** stamp and **Keep** pressed.
- Change keepers with that button or the keyboard. Marks are remembered on this computer.
- Save marked keepers as copies or hard links, write XMP sidecars, or save a JSON and CSV manifest.

## Install

You need Python 3.11 or newer. Outburst runs on macOS (including Apple Silicon), Windows, and Linux. On Apple Silicon, use an arm64 build of Python.

| System | Double-click |
| --- | --- |
| macOS | `Outburst.command` |
| Windows | `Outburst.bat` |
| Linux | `Outburst.desktop`, or run `./outburst.sh` |

The first run creates `.venv`, installs the libraries (including subject and bird-eye detection), and opens the window. That first install downloads PyTorch and can take several minutes. Later runs open the window. An older environment installs the extra the next time you start the launcher, or the first time you turn either feature on.

```bash
./outburst.sh examples/sample_burst
```

```bat
Outburst.bat examples\sample_burst
```

### Install by hand

macOS and Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[subject]"
outburst
```

Windows:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -e ".[subject]"
outburst
```

Open a folder immediately with `outburst examples/sample_burst`. Redraw the sample with `python examples/generate_sample_burst.py`.

### exiftool

Install [exiftool](https://exiftool.org/) if you can. Outburst uses it first, including for CR3.

- macOS: `brew install exiftool`
- Windows: the ExifTool Windows package. Outburst starts it without a console window.
- Linux: `libimage-exiftool-perl`, or the equivalent for your distribution

Without exiftool, the bundled exifread library reads JPEG and TIFF-based RAW (CR2, NEF, ARW, DNG, ORF, PEF, RW2). CR3, RAF, and X3F fall through to a date-string search. A frame with no capture time stays in a burst by itself. Opening a RAW folder without exiftool names the formats in that folder.

If the window fails to open because `libEGL.so.1` is missing, install `libegl1` (Debian and Ubuntu) and try again.

## The window

**Open** reads the folder you pick. Check **Include subfolders on the next scan** to go deeper. Supported RAW types include `CR2` `CR3` `NEF` `NRW` `ARW` `ORF` `RAF` `RW2` `DNG` `PEF` `SRW`, plus the other extensions in `src/outburst/scanner.py`, and JPEG. A JPEG beside a RAW file of the same name is that frame's preview.

The first visit reads each file. Later visits are faster, because measurements and thumbnails stay in a cache on this computer. **Rescan** drops that cache for the open folder and reads it again. The list, preview, and filmstrip stay empty until scoring finishes. Keeper marks stay. **Clear cache** in the View menu forgets every folder.

The left side is **Bursts**, **Grouping**, and **Scoring**. Grouping starts collapsed and Scoring starts open, and that choice is remembered. The open folder sits on the right of the toolbar. Click the path to open that folder. **Rescan** is still the toolbar action. Under the preview, stars set the rating, and **Keep** and **Reject** set the pick and reject flags. The buttons stay labeled **Keep · A** and **Reject · X**. Green or red fill shows which one is on. Reject only flags the frame: the photograph stays where it is, and Save keepers skips it. A star rating on a rejected frame is still written to a sidecar. The filmstrip shows the burst, scored 0 to 100. The selected card's score is larger. **Time** follows the shutter. **Score** puts the highest score on the left, with ties left in shooting order. Arrow keys, Home, and End follow the order on screen. **CLIP** is a badly blown or crushed exposure. **SOFT** is well behind the best frame in that burst. While **Minimum score** is on, the control shows how many frames are still on screen, such as “1 of 5 showing.”

**Burst gap** is how close two frames must be to stay together. The default is 1.0 second. A gap of exactly 1.0 second starts a new burst. Moving it regroups immediately. Keepers you marked stay. Outburst clears its own pick and marks the best frame of each burst.

**Sharp** and **Center** change the ranking without reading the files again. Sharpness is three quarters of the score. The center of the frame counts more than the corners.

**Info**, beside the score, starts off. It shows the lines after the Laplacian number: the frame, the exposure, and the other measurements.

**Z** shows the preview at 1:1. **+**, **−**, and the scroll wheel step through fit, 1:1, 2:1, and 4:1. Drag to look around. **F** paints the strongest edges green. These use the preview Outburst scored.

### Keyboard

These work unless a slider or the burst-gap box is focused. Arrow keys then adjust that control.

| Keys | Action |
| --- | --- |
| Right / Left | Next or previous frame, in filmstrip order |
| Down / Up | Next or previous burst |
| Home / End | First or last frame, in filmstrip order |
| Enter | Jump to the recommended frame |
| C | Compare. Starts beside the best other frame |
| [ / ] | Move the other side of compare |
| Shift-click | Compare with that thumbnail |
| Z | Zoom to 1:1. Press again to fit |
| + / − | Zoom through fit, 1:1, 2:1, and 4:1 |
| F | Focus peaking |
| 1–5 | Set that star rating. Press the same number again to clear it |
| 0 | Clear the star rating |
| A | Keep the current frame. Press again to clear |
| X | Reject this frame. The file stays put, and Save keepers skips it. Press again to clear |
| R | Reject the burst. Press again to clear |
| G | Reject obvious misses. Press again to restore them |
| Ctrl+O | Open a folder |
| F5 | Rescan this folder |
| Ctrl+E | Save keepers |
| Ctrl+Shift+E | Save a manifest |
| Ctrl+Shift+X | Write XMP sidecars |
| Esc | Cancel the scan, download, or install |

## How a frame gets its score

Outburst scores the embedded JPEG preview, or the JPEG that sits beside a RAW file of the same name. The preview is scaled so the long side is at most 1024 pixels. The RAW file is not demosaiced. Each frame is scored on its own: one frame's number does not depend on the others in the burst.

The filmstrip shows that score times 100, rounded to a whole number from 0 to 100.

### Sharpness

Sharpness is how much fine detail the preview has. Outburst turns the preview gray and measures the variance of a Laplacian. A crisp edge spreads those values. A soft frame does not.

Two sharpness numbers are kept:

- The whole frame.
- The middle 60 percent of the width and of the height. Inside that middle, nine patches are measured. The patch number is 85 percent of the sharpest patch plus 15 percent of the next, so one bright speck does not decide it.

**Center** mixes those two. The default is 65, so the score uses 65 percent of the middle patch and 35 percent of the whole frame. At 100, only the middle patch counts, and grass or branches in the corners drop out. At 0, only the whole frame counts. Moving Center rescores immediately from measurements already saved. The files are not read again.

Edge energy, shown under **Info**, is a second edge measurement (Tenengrad). It is stored for you to read. It is left out of the score and out of the ranking.

### Exposure

Exposure is read from the preview, not from a RAW histogram. Pixels at or above 250 count as clipped highlights. Pixels at or below 5 count as crushed shadows. A little specular highlight is tolerated. A large blown area costs more than crushed shadows, and a frame far from a mid gray loses a little more. The result is a number from 0 to 1.

### The number on the filmstrip

**Sharp** is the share of the score that comes from sharpness. The rest is exposure. The default is 75, so sharpness is three quarters and exposure is one quarter. At 100 the score is sharpness only. At 0 it is exposure only. Moving Sharp rescores immediately, without reading the files again.

Before the mix, sharpness is squeezed into the range 0 to 1 by dividing it by itself plus 300. A very sharp frame approaches 1. A soft frame stays low. That scaled sharpness and the exposure number are then mixed by **Sharp**. Multiply the mix by 100 and you have the filmstrip number.

### Which frame is recommended

Inside a burst, the highest score is the recommendation. If two scores match, the sharper whole frame wins, then the earlier filename. When a scan finishes, that frame is marked a keeper when the burst does not already have one.

**Subject-aware sharpness** changes which sharpness number is used. When it is on and an animal is found, the sharpest patch inside that box, expanded by 12 percent, replaces the Center blend for that frame. **Center** does not apply to a frame that has a measured animal. When nothing is found, that frame keeps the Center blend. Turning the checkbox on scans the folder again and downloads the selected model when the file is missing. Turning it off rescores from the measurements already saved. Choosing another model can move the box, and that can change the score. **Show subject box** only hides the rectangle. The score still uses the box.

**Bird eye detection** leaves the 0 to 100 score as it is and can change which frame is recommended. When two or more frames in the burst have an eye large enough to rate, the sharpest of those eyes becomes the recommendation, and an earlier filename breaks a tie. An eye is rated when the disk on the analysis preview is at least 16 pixels across. A smaller eye is noted and left out of the ranking. One rated eye, or none, leaves the usual score ranking in place. Turning the checkbox on or off scans the folder again. **Show eye circle** only hides the ring.

### What leaves the score alone

**Burst gap** does not change a frame's number. It decides which frames share a burst, and so which frames compete for the recommendation. The default is 1.0 second. Frames closer than that stay together. A gap of exactly 1.0 second starts a new burst. Moving it regroups immediately. Keepers you marked stay. Outburst clears the keeper it chose and marks the best frame of each new burst.

**Minimum** hides bursts shorter than the number you set. The default is 2 frames. Their scores stay. They leave the list and are left out of Save.

**Minimum score** hides frames whose on-screen number is below the cutoff. A burst with none left leaves the list. The scores, the marks, and Save keepers stay as they are.

**Info** only shows or hides the extra measurements.

## Your files

Scanning writes only into the cache:

| System | Cache |
| --- | --- |
| macOS | `~/Library/Caches/Outburst` |
| Windows | `%LOCALAPPDATA%\Outburst\Cache` |
| Linux | `~/.cache/outburst` |

Delete that folder to measure everything again. `outburst.log` is kept there too.

**Save** copies the frames you marked into a folder you pick and writes an `.xmp` sidecar beside each saved keeper. After a scan, each burst's recommended frame is marked when that burst has no keeper yet. **G** rejects the other frames that are much softer, or heavily clipped, and leaves your keepers. A burst with no keeper is skipped. A burst shorter than **Minimum** is hidden and is not saved. Hard links are offered when you want a second name for the same file. If a link cannot be created, Outburst copies the file. Deleting the copy leaves the original.

**XMP** writes a sidecar next to the original for each keeper and each star-rated frame. Lightroom, Capture One, Darktable, and digiKam can read it. Save writes that same sidecar into the keepers folder.

- Stars are `xmp:Rating` from 0 to 5, and `MicrosoftPhoto:Rating` as the matching percent.
- A pick is `xmpDM:pick="1"` and `xmpDM:good="true"`. A pick with no stars of its own is also written as 5 stars.
- A reject is `xmpDM:pick="-1"` and `xmpDM:good="false"`. With no stars of its own, `xmp:Rating` is -1. Stars you set stay beside that flag.

A burst shorter than the minimum is not saved, so a pick on one of those frames is not written. Stars on that frame still are. A reject with no stars gets no sidecar. The sidecar also stores `outburst:Score`, the burst id, and whether the frame was the recommendation. Outburst asks before writing. An existing XMP file is updated in Outburst's own fields. A file that is not XMP is left alone.

**Save manifest** writes `outburst-report.json` and `outburst-report.csv`: every frame, its score, and the keeper marks. `examples/sample_output` shows the shape. The photographs stay as they are.

## Subject detection

This stays off until you check **Subject-aware sharpness**. The model list is YOLO11 Nano (the default), YOLO11 Small, YOLOv8 Nano, and YOLOv8 Small. Checking the box, or choosing a model, scores the open folder and downloads the weights when the file is missing. Cancel with Esc. Unchecking the box scores again from the measurements already cached.

When an animal is found, the sharpest patch inside that box becomes the score, and the box is drawn. **Show subject box** hides the rectangle without scanning again. The score still uses the box, and zoom still centers on it when no eye is present. If nothing is found, Outburst keeps the center-patch score and says so.

From this project, weights go in `models` next to this README. Otherwise they go in `models` inside the cache. Set `OUTBURST_MODELS` to use another folder. **ready** means the file is already there. **download** means it is fetched when you choose it. Photos stay on this computer.

On Apple Silicon the subject detector uses MPS when PyTorch provides it. On NVIDIA it uses CUDA. Otherwise it uses the CPU.

## Bird eye detection

This stays off until you check **Bird eye detection**. Checking it, or unchecking it, scans the open folder again. The first time, Outburst downloads `eye_pose_v0.pt` (about 6 MB) into the models folder. That file is the open-weight [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0) model, a YOLO11n pose network trained on CUB-200-2011 bird photos. The weights are AGPL-3.0. Photos stay on this computer. The eye finder stays on the CPU.

The model is birds only, so a distant or unusual field frame may have no eye. Outburst checks the whole preview, then overlapping 512 pixel windows, and maps the strongest eye back. A keypoint under 0.35 confidence is ignored. An eye is rated only when the disk around it is at least 16 pixels across. A smaller eye is noted and left out of the ranking.

With two or more rated eyes in a burst, the sharpest eye is the best frame and Outburst marks it as a keeper. A reject on that frame stays rejected. A keeper you already set stays yours. A burst with fewer than two rated eyes keeps the usual sharpness rank.

A detected eye is a gold ring labeled EYE. **Show eye circle** hides the ring without scanning again. The score and the zoom center stay. **Info** shows the eye sentence with the other measurements. The filmstrip says EYE. The burst row and the status bar count detected eyes, including frames in bursts hidden by **Minimum**.

In an older environment that does not yet have Ultralytics, turning **Subject-aware sharpness** or **Bird eye detection** on installs it before the scan. The status bar says that PyTorch is downloading. You can also run `pip install "outburst[subject]"`.

## For developers

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On Windows, activate with `.venv\Scripts\activate`. See [docs/architecture.md](docs/architecture.md) for the scoring formula, cache keys, and which modules write files. The launcher is the supported way to install and open Outburst.

Later versions can add a bundled small detector and signed installers for macOS and Windows.

## License

[MIT](LICENSE).
