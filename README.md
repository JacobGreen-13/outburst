# Outburst

Outburst culls wildlife and nature bursts on your own computer. It groups frames shot a fraction of a second apart, scores each one, and opens on the frame it would keep.

Nothing is uploaded. There is no account. Original RAW and JPEG files stay untouched. Subject and bird eye models stay off until you install that software, and each weight file is downloaded when that file is missing.

## Contents

- [What you can do](#what-you-can-do)
- [Requirements](#requirements)
  - [The app](#the-app)
  - [Subject and bird eye detection](#subject-and-bird-eye-detection)
- [Install](#install)
  - [Windows 11](#windows-11)
  - [Debian and Ubuntu](#debian-and-ubuntu)
  - [From source](#from-source)
  - [Install by hand](#install-by-hand)
  - [exiftool](#exiftool)
- [The window](#the-window)
  - [Keyboard](#keyboard)
- [How a frame gets its score](#how-a-frame-gets-its-score)
  - [Sharpness](#sharpness)
  - [Exposure](#exposure)
  - [The number on the filmstrip](#the-number-on-the-filmstrip)
  - [Which frame is recommended](#which-frame-is-recommended)
  - [What leaves the score alone](#what-leaves-the-score-alone)
- [Your files](#your-files)
- [Subject detection](#subject-detection)
- [Bird eye detection](#bird-eye-detection)
- [For developers](#for-developers)
- [Feedback](#feedback)
- [License](#license)

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

## Requirements

### The app

The Windows setup program, the Debian package, and a source checkout have different requirements. Culling itself stays on this computer. Once the app is installed, opening a folder and scoring bursts works with the computer offline.

**Windows 11 setup program**

- Windows 11, 64-bit, on an Intel or AMD processor.
- The setup program is [Outburst-Setup-0.2.1.exe](https://github.com/JacobGreen-13/outburst/releases/tag/v0.2.1) on the Releases page. It is about 80 MB.
- It installs for the current user under `%LOCALAPPDATA%\Outburst` and adds a Start menu shortcut. ExifTool is included beside the app. Open **Outburst** from the Start menu after setup.
- The setup program is unsigned. The first time Windows SmartScreen appears, choose **More info**, then **Run anyway**.

**Debian and Ubuntu package**

- Debian 12 or newer, or Ubuntu 22.04 or newer, 64-bit, on Intel, AMD, or ARM.
- The packages are [outburst_0.2.1_amd64.deb](https://github.com/JacobGreen-13/outburst/releases/tag/v0.2.1) and [outburst_0.2.1_arm64.deb](https://github.com/JacobGreen-13/outburst/releases/tag/v0.2.1) on the Releases page.
- The package installs under `/opt/outburst`, adds **Outburst** to the application menu, and uses the ExifTool package from the distribution.
- PyTorch and the model files stay in the home directory, so installing a newer package keeps a detection setup that is already there.

**From source**

- Python 3.11 or newer.
- macOS, Windows, or Linux. On Apple Silicon, use an arm64 build of Python.
- Internet the first time, so the launcher can create `.venv` and install Outburst.
- [exiftool](https://exiftool.org/) is recommended, especially for CR3. The Windows setup program includes it, and the Debian package installs it from the distribution.

### Subject and bird eye detection

These are optional. Until they are installed, every frame is scored from the whole preview and the center blend. The controls stay hidden, and the checkboxes stay off after the install until you turn them on.

**Computer**

- 64-bit Windows 10 or Windows 11 on an Intel or AMD processor, with an NVIDIA GPU. The setup program itself is Windows 11.
- 64-bit Linux on Intel, AMD, or ARM, with an NVIDIA GPU.
- An Apple Silicon Mac running macOS 14 or newer.
- At least 5 GB free where the detection software will be installed.
- An internet connection while the software and any missing model files are downloading.

The install button checks the computer before it offers the license. It stops when the computer is 32-bit, an Intel Mac, Windows on ARM, a Windows or Linux computer without an NVIDIA GPU, or another system outside that list. It also stops when free space is under 5 GB, or when the internet check fails. In each of those cases Outburst explains why, leaves the controls hidden, and leaves the model files where they are. The license is offered only after those checks pass.

**Software the button adds**

- [Ultralytics](https://github.com/ultralytics/ultralytics) 8.3 or newer, under the [GNU AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html). Its license file is [LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).
- [PyTorch](https://github.com/pytorch/pytorch), which Ultralytics installs. The download is large and can take several minutes. PyTorch is [BSD-3](https://github.com/pytorch/pytorch/blob/main/LICENSE).
- On the Windows setup program, a private Python 3.12 under `%LOCALAPPDATA%\Outburst\detection`. Detection runs in that Python. The photographs stay on this computer.
- On the Debian package, a user-owned Python environment under `~/.local/share/Outburst/detection`. Detection runs in that environment. The photographs stay on this computer.
- From source, Ultralytics and PyTorch go into the same Python that is running Outburst.

You accept the AGPL-3.0 license in the app before any of that is downloaded. Declining leaves the controls hidden.

**Model files**

All five weight files are AGPL-3.0. The install button downloads YOLO11 Nano and `eye_pose_v0.pt` into the models folder after the software is in place. The other subject models download when you choose them. The Help menu opens the same pages.

| Model | File | About | Use | Source |
| --- | --- | --- | --- | --- |
| YOLO11 Nano | `yolo11n.pt` | 6 MB | Default subject model | [weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) |
| YOLO11 Small | `yolo11s.pt` | 19 MB | More accurate subject model | [weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt) |
| YOLOv8 Nano | `yolov8n.pt` | 6 MB | Subject model | [weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt) |
| YOLOv8 Small | `yolov8s.pt` | 22 MB | More accurate subject model | [weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt) |
| eye-pose-v0 | `eye_pose_v0.pt` | 6 MB | Bird eyes | [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0) |

The four subject files come from the [Ultralytics assets v8.3.0 release](https://github.com/ultralytics/assets/releases/tag/v8.3.0). `eye_pose_v0.pt` is the open-weight [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0) model, a YOLO11n pose network trained on CUB-200-2011 bird photos. The direct file is [eye_pose_v0.pt](https://huggingface.co/synthet/eye-pose-v0/resolve/main/eye_pose_v0.pt).

Subject detection uses MPS on Apple Silicon when PyTorch provides it, and CUDA when an NVIDIA GPU is present. Bird eye detection uses CUDA on NVIDIA and the CPU on a Mac. A scan can fall back to the CPU when the GPU cannot be used after the software is installed.

## Install

### Windows 11

1. Download [Outburst-Setup-0.2.1.exe](https://github.com/JacobGreen-13/outburst/releases/tag/v0.2.1).
2. If SmartScreen appears, choose **More info**, then **Run anyway**.
3. Finish the setup. Open **Outburst** from the Start menu. A newer setup keeps PyTorch, Ultralytics, and the model files from the previous install, and the detection controls appear when those are already on this computer.
4. In **Scoring**, press **Install subject and bird eye detection**.
5. Outburst checks the computer, then the internet connection. A failed check explains itself and stops there.
6. Read the AGPL-3.0 notice and choose **Install**.
7. Wait while the private Python, Ultralytics, and PyTorch download, then YOLO11 Nano and `eye_pose_v0.pt`. The status bar names the step and shows how many megabytes have arrived. PyTorch is a large download, so that count keeps climbing for several minutes. **Esc** cancels.
8. The subject and bird eye controls appear. Both checkboxes stay off.
9. Check **Subject-aware sharpness** to score inside an animal when one is found. YOLO11 Nano is already downloaded and is the default. Choose another model in the list when you want it. That choice downloads the file when it is missing.
10. Check **Bird eye detection** to rank a burst by the sharpest rated eye when two or more frames have one. `eye_pose_v0.pt` is already downloaded when the install finished.
11. **Show subject box** and **Show eye circle** only hide the drawing. The score and the recommendation stay.

To build that setup program on a Windows machine:

```powershell
powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

The result is `dist\installer\Outburst-Setup-0.2.1.exe`. GitHub Actions builds the same file with [`.github/workflows/windows-installer.yml`](.github/workflows/windows-installer.yml). Inno Setup 6 is required when you build it by hand. `choco install innosetup` provides it.

### Debian and Ubuntu

1. Download `outburst_0.2.1_amd64.deb` for Intel or AMD, or `outburst_0.2.1_arm64.deb` for ARM, from the [Releases](https://github.com/JacobGreen-13/outburst/releases/tag/v0.2.1) page.
2. Install it. Apt installs the libraries the package needs, including ExifTool:

```bash
sudo apt install ./outburst_0.2.1_amd64.deb
```

3. Open **Outburst** from the application menu, or run `outburst`. A newer package keeps PyTorch, Ultralytics, and the model files already in the home directory, and the detection controls appear when those are already on this computer.
4. In **Scoring**, press **Install subject and bird eye detection**. The rest of the steps match the Windows install: the computer check, the internet check, the AGPL-3.0 notice, then Ultralytics, PyTorch, YOLO11 Nano, and `eye_pose_v0.pt`.

To build the package on Debian or Ubuntu:

```bash
packaging/linux/build.sh
```

The result is `dist/installer/outburst_0.2.1_amd64.deb` on Intel or AMD, and `dist/installer/outburst_0.2.1_arm64.deb` on ARM. GitHub Actions builds both with [`.github/workflows/debian-package.yml`](.github/workflows/debian-package.yml). The script reads the window libraries from the packaged binaries.

### From source

| System | Double-click |
| --- | --- |
| macOS | `Outburst.command` |
| Windows | `Outburst.bat` |
| Linux | `Outburst.desktop`, or run `./outburst.sh` |

The first run creates `.venv`, installs Outburst, and opens the window. Subject and bird eye controls stay hidden. **Install subject and bird eye detection** follows the same computer check, internet check, and AGPL-3.0 acceptance as the Windows app. Accepting installs Ultralytics and PyTorch into that `.venv`, then downloads the models. The OpenCV already loaded by Outburst stays in place, so the install can finish while the window is open. The status bar names the step and shows how many megabytes have arrived. PyTorch is a large download, so that count keeps climbing for several minutes. The detection controls appear after the software and both model files are in place. A newer copy uses PyTorch and the model files already on this computer, including the private Python from the Windows setup and a `.venv` beside the program. That Python runs the detection code shipped with this version. The checkboxes stay off until you turn them on.

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
pip install -e .
outburst
```

Windows:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
pip install -e .
outburst
```

That installs the app. Detection stays on the **Install subject and bird eye detection** button. That button is what checks the computer, asks for the AGPL-3.0 license, installs Ultralytics and PyTorch, and downloads YOLO11 Nano and `eye_pose_v0.pt`. Opening a folder, checking a box, or scanning does not do those steps. To put Ultralytics and PyTorch into the environment before you open the window, use `pip install -e ".[subject]"`. The model files still come from the install button. Choosing another subject model after that downloads that file.

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

**Subject-aware sharpness** changes which sharpness number is used. When it is on and an animal is found, the sharpest patch inside that box, expanded by 12 percent, replaces the Center blend for that frame. **Center** does not apply to a frame that has a measured animal. When nothing is found, that frame keeps the Center blend. The checkbox is hidden until **Install subject and bird eye detection** finishes. That button checks the computer and the internet connection, asks you to accept the AGPL-3.0 license, installs the detection software, and downloads the models. An incompatible computer or a missing connection stays on whole-frame scoring, with an explanation, and nothing is installed or downloaded. Turning the checkbox off rescores from the measurements already saved. Choosing another model can move the box, and that can change the score. **Show subject box** only hides the rectangle. The score still uses the box.

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

In **Scoring**, press **Install subject and bird eye detection**. Outburst checks the computer for an NVIDIA GPU or Apple Silicon, then the internet connection, and only then asks you to accept the AGPL-3.0 license. Accepting installs Ultralytics and PyTorch and downloads YOLO11 Nano and `eye_pose_v0.pt`. Declining leaves this section as that button. An incompatible computer or a missing connection explains why and leaves the model files where they are.

When the install finishes, the controls appear and **Subject-aware sharpness** stays off until you check it. The model list is YOLO11 Nano (the default), YOLO11 Small, YOLOv8 Nano, and YOLOv8 Small. The files, sizes, license, and source pages are in [Requirements](#requirements). Checking the box scores the open folder. Choosing another model downloads that file when it is missing, then scores the open folder. Cancel with Esc. Unchecking the box scores again from the measurements already cached.

When an animal is found, the sharpest patch inside that box becomes the score, and the box is drawn. **Show subject box** hides the rectangle without scanning again. The score still uses the box, and zoom still centers on it when no eye is present. If nothing is found, Outburst keeps the center-patch score and says so.

From this project, weights go in `models` next to this README. Otherwise they go in `models` inside the cache. Set `OUTBURST_MODELS` to use another folder. **ready** means the file is already there. **download** means it is fetched when you choose it. Photos stay on this computer.

On Apple Silicon the subject detector uses MPS when PyTorch provides it. On NVIDIA it uses CUDA. Otherwise it uses the CPU.

## Bird eye detection

**Bird eye detection** is in the same Scoring section as subject detection, and it stays hidden until you press **Install subject and bird eye detection**. That button checks the computer for an NVIDIA GPU or Apple Silicon, then the internet connection, and only then asks you to accept the AGPL-3.0 license. Accepting installs Ultralytics and PyTorch and downloads `eye_pose_v0.pt` (about 6 MB) along with YOLO11 Nano. The checkbox stays off until you check it.

Checking it, or unchecking it, scans the open folder again. If `eye_pose_v0.pt` is missing, that check downloads it into the models folder. The file is the open-weight [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0) model, a YOLO11n pose network trained on CUB-200-2011 bird photos. The weights are [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html). Photos stay on this computer. The eye finder stays on the CPU.

The model is birds only, so a distant or unusual field frame may have no eye. Outburst checks the whole preview, then overlapping 512 pixel windows, and maps the strongest eye back. A keypoint under 0.35 confidence is ignored. An eye is rated only when the disk around it is at least 16 pixels across. A smaller eye is noted and left out of the ranking.

With two or more rated eyes in a burst, the sharpest eye is the best frame and Outburst marks it as a keeper. A reject on that frame stays rejected. A keeper you already set stays yours. A burst with fewer than two rated eyes keeps the usual sharpness rank.

A detected eye is a gold ring labeled EYE. **Show eye circle** hides the ring without scanning again. The score and the zoom center stay. **Info** shows the eye sentence with the other measurements. The filmstrip says EYE. The burst row and the status bar count detected eyes, including frames in bursts hidden by **Minimum**.

## For developers

```bash
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

On Windows, activate with `.venv\Scripts\activate`. See [docs/architecture.md](docs/architecture.md) for the scoring formula, cache keys, and which modules write files. The launcher is the supported way to install and open Outburst from source. The Windows 11 setup program and the Debian package are on the [Releases](https://github.com/JacobGreen-13/outburst/releases/tag/v0.2.1) page.

## Feedback

A note from a real cull is welcome. If a score keeps the wrong frame, Outburst cannot open a photograph, or the detection install stops, open an [issue](https://github.com/JacobGreen-13/outburst/issues). The useful details are the camera, the operating system, and the version in **Help → About Outburst**.

## License

Outburst itself is [MIT](LICENSE).

Subject detection and bird eye detection are optional. Installing them brings in software and weight files under their own licenses. The Help menu opens these pages.

| Piece | License | Source |
| --- | --- | --- |
| Outburst | [MIT](LICENSE) | This repository |
| Ultralytics | [AGPL-3.0](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) |
| YOLO11 and YOLOv8 weights | [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) | [Ultralytics assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0) |
| synthet/eye-pose-v0 | [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) | [synthet/eye-pose-v0](https://huggingface.co/synthet/eye-pose-v0) |
| PyTorch | [BSD-3](https://github.com/pytorch/pytorch/blob/main/LICENSE) | [pytorch/pytorch](https://github.com/pytorch/pytorch) |
