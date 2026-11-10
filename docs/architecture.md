# Outburst architecture

Outburst is a desktop app. The scoring pipeline does not import Qt, so it can be tested and reused without a window. The window is a client of that pipeline.

## Layout

| Module | Role |
| --- | --- |
| `scanner.py` | Finds RAW and JPEG files. A JPEG with the same stem as a RAW in the same folder is the preview pair, not a second frame. |
| `metadata.py` | Reads `DateTimeOriginal` and `SubSecTimeOriginal`. Uses exiftool when it is installed, starting it with no console window on Windows. Otherwise it uses exifread for JPEG and TIFF-based RAW, then a built-in TIFF/JPEG reader, then a date string in the file header. exifread does not recognize CR3, RAF, or X3F. |
| `grouping.py` | Sorts by capture time and groups consecutive frames whose gap is strictly below the threshold. Frames with no capture time stay alone. |
| `preview.py` | Loads a preview. Paired JPEGs win. Otherwise rawpy's embedded thumbnail is used. A byte scan for an embedded JPEG is the fallback. Full demosaic is never requested. |
| `scoring.py` | Laplacian variance, optional Tenengrad, exposure penalties, and the weighted composite. Pure functions. |
| `eyes.py` | Optional bird eye detection. `eye-pose-v0` supplies the eye keypoints from a BGR preview, then from overlapping windows when the full preview has no eye. A disk is rated only when it is at least 16 px across on the analysis preview. The pose model runs on CPU. |
| `subject.py` | Optional YOLO detector and the model catalog. Device order is Apple Silicon MPS, then NVIDIA CUDA, then CPU. `subject_hardware_support` checks the computer before an install. `internet_connection` checks the network before that install or a model download. `ensure_model` downloads a missing weights file into the models folder. `install_subject_runtime` installs Ultralytics into the running Python after the license is accepted. `load_yolo_detector` only opens a file that is already there. |
| `cache.py` | SQLite plus JPEG previews in the user cache directory. |
| `pipeline.py` | Ties the steps together and rebuilds scores or groups when weights or the gap change. |
| `output.py` | XMP sidecars, keeper copies or hard links, JSON and CSV reports. |
| `gui/` | PySide6 window, filmstrip, preview, and the background scan thread. |

## Pipeline

1. List the folder. Hidden files and hidden directories are skipped. Subfolders are included only when the checkbox is on.
2. Reuse a cache row when the path, size, modification time, score version, and subject-model key match and the preview files are still on disk.
3. For cache misses, read capture time, then score the misses with a pool of two to four workers. Each worker decodes a preview and shrinks the long edge to 1024 px before measuring. A 360 px thumbnail is stored for the filmstrip. The scan thread writes the SQLite row, in file order, after the worker finishes that frame.
4. If a subject detector or the bird-eye model is active, workers decode ahead and the scan thread runs those models on one batch of up to four frames at a time. The same workers then measure sharpness. Neither model is shared across threads.
5. If bird eye detection is on, the pose model reads the left and right eye keypoints from the BGR preview. A preview with no eye is searched again in overlapping 512 px windows. The center, a radius taken from the bird box, and the sharpness (only when the disk is large enough) are stored on the metrics. The cache key includes the weights file and an eye-reading token, so a scan without the model, or from an older reading, does not reuse those rows. Turning it on with no weights file leaves detection off and adds a warning.
6. Group the frames and hand a `Library` to the window.

Changing the sharpness, exposure, or center sliders recomputes the composite from cached measurements. Changing the burst gap only regroups. Neither step rereads the RAW files. Turning subject detection or bird eye detection on starts a new scan, because that measurement is not in the plain cache row. Turning bird eye detection off does the same, and clears automatic keepers first. The bird-eye weights download the same way as a subject model.

## Scoring

Measurements:

- **Sharpness** is the variance of a Laplacian on the grayscale preview. Higher means more fine detail.
- **Center sharpness** uses the middle 60% of width and height.
- **Critical sharpness** is the sharpest patch inside that center crop. Nine windows cover the crop. The value is 85% of the sharpest window plus 15% of the next, so one speck does not own the score.
- **Tenengrad** (mean squared Sobel gradient) is stored and shown as edge energy. It does not affect the rank.
- **Exposure** is 0 to 1. Pixels at or above 250 count as clipped highlights. Pixels at or below 5 count as crushed shadows. Highlights are penalized more heavily. A frame far from a mid tone is penalized a little.
- **Subject sharpness**, when a box exists, is the sharpest patch inside that box expanded by 12%.

The sharpness that enters the score is:

```text
center_mix * critical_sharpness + (1 - center_mix) * sharpness
```

Older cached rows have no critical sharpness. Those rows use center sharpness in that same blend until a rescan. `SCORE_VERSION` is 2, so a new scan rebuilds them.

If subject scoring is on and a subject was found, the subject patch replaces the blend.

It is then mapped to 0..1 with a fixed scale so one frame does not depend on the rest of the burst:

```text
normalized = sharpness / (sharpness + 300)
composite  = (sharpness_weight * normalized + exposure_weight * exposure)
             / (sharpness_weight + exposure_weight)
```

The window shows `composite * 100` as a whole number. The recommended frame is the highest composite. Ties prefer a higher full-frame Laplacian, then the earlier filename.

When two or more frames in a burst have `eye_sharpness`, the recommended frame is the highest eye sharpness instead. An earlier filename breaks that tie. One rated eye, or none, leaves the composite rank in place. Eye sharpness is the Laplacian variance inside the eye circle from the pose model. A detected eye smaller than 16 px is stored without a sharpness value.

`SCORE_VERSION` in `scoring.py` is part of the cache key. Bump it when the measurement formula changes.

## Cache and decisions

The database lives in the user cache directory (`~/Library/Caches/Outburst` on macOS, `%LOCALAPPDATA%\Outburst\Cache` on Windows, `$XDG_CACHE_HOME/outburst` or `~/.cache/outburst` on Linux). Keeper and reject marks are rows in the same database, keyed by the photograph's path. A keep row also records whether Outburst chose it (`auto`) or the photographer did (`user`). Rows written before that column existed are treated as the photographer's marks. Decisions survive a restart and a cache clear. They are not written into the photograph.

`Cache.forget_folder` deletes measurement rows and their preview files for paths inside one folder. **Rescan** calls it before reading that folder again. `Cache.clear` does the same for every folder. Neither call deletes decision rows.

## What is allowed to be written

| Action | Where it writes |
| --- | --- |
| Scan and score | Cache directory only |
| Keep / reject | Cache directory only |
| Write XMP | A new or updated `<stem>.xmp` beside a keeper or a star-rated frame, and only after confirmation. Other frames are left alone. Stars go to `xmp:Rating` and `MicrosoftPhoto:Rating`. A pick is `xmpDM:pick` 1 and `digiKam:PickLabel` 3. A starred reject is `xmpDM:pick` -1 and `digiKam:PickLabel` 1. A reject with no stars gets no sidecar |
| Save keepers | A folder the photographer chooses. Copy, or a hard link with a copy fallback, plus an `.xmp` sidecar beside each saved keeper. Only marked frames from bursts that meet Minimum frames. Hidden shorter bursts are not saved |
| Save manifest | `outburst-report.json` and `outburst-report.csv` in a chosen folder. Every frame, its score, and the keeper marks. Photos are not changed |
| Download a subject model | The models folder: `models` in this project when Outburst is run from the checkout, `models` inside the cache directory otherwise, or `OUTBURST_MODELS` when that is set |

Photograph files are opened for reading. Export uses `shutil.copy2` or `os.link`. If an XMP sidecar exists and does not look like XMP, Outburst leaves it alone and reports the file.

## Window

The window is a toolbar, a left rail, the preview, and the filmstrip. The toolbar holds Open, Rescan, Save, Save manifest, and XMP, and the open folder path on the right. Clicking that path opens the folder. The rail is Bursts plus Grouping and Scoring. Grouping starts collapsed and Scoring starts open, and both choices are saved. Grouping is the burst gap, the minimum length, and subfolders. Scoring is the sharpness and center sliders, an optional minimum score, the subject and bird-eye toggles, and the model. Minimum score hides frames below that 0 to 100 display score and shows how many remain, such as “1 of 5 showing,”, and hides a burst when none of its frames remain. Keeper marks on those frames are kept and can still be saved. The filmstrip is one frame with Time and Score at its top right. `AnalyzeWorker` is a `QThread`. It emits progress and a finished `Library`. While that scan runs, the burst list, preview, and filmstrip are cleared so the previous frames cannot be marked or compared. Preview scoring inside that thread uses two to four workers. **Save keepers** and **Write XMP sidecars** each run on their own thread and write two files at a time, with progress in the status bar. The Save and sidecar actions stay disabled until that write finishes. The window ignores results from a scan that was cancelled or replaced, using a scan id. `ModelFetchWorker` is a separate thread for the weights download. The burst list hides bursts shorter than **Minimum frames** (2 by default). When a library is shown, each burst that has no keeper yet gets its recommended frame marked keep, stored as `auto`. If the burst has two or more rated eyes, that mark follows the sharpest eye: other automatic keepers in the burst are cleared, a reject is left rejected, and a keeper the photographer set keeps its origin. Changing the burst gap deletes automatic keeps, leaves `user` keeps, and marks the best frame of each regrouped burst as `auto` unless that frame is rejected or already a user keep. The **Eye sharpness** checkbox does the same clear, then rescans. **Save keepers** writes marked frames from the bursts that are showing and skips the hidden shorter ones. The preview has **Keep** and **Reject** buttons, labeled with A and X. The labels stay put, and the selected filmstrip score is larger than the others. Scoring has **Show subject box** and **Show eye circle**, separate from the detection checkboxes. Both start on. Hiding a mark redraws the preview and does not rescan. The score and the zoom center stay unchanged. **Info**, beside the score, starts off and shows the lines after the Laplacian number. **C** compares the current frame with another frame in the burst. It starts on the best other frame, or the next sharpest when the current frame is already the best. **[** and **]** move that other frame, and Shift-click on a thumbnail chooses it. **Z** shows the cached preview at 1:1. **+**, **−**, and the scroll wheel step through fit, 1:1, 2:1, and 4:1, and drag pans the zoomed preview. **F** paints the strongest edges. **G** rejects frames that trail the best score by 0.18 or more, or whose highlight clip is at least 8% or shadow clip at least 25%. It skips the recommended frame and any keeper. Pressing **G** again clears the rejects that press just made. **Time** and **Score** above the filmstrip reorder that burst. Time is shooting order. Score puts the highest composite first, keeps ties in shooting order, and leaves a frame with no score at the right. Left, right, Home, and End follow the order on screen. The choice is stored as `filmstrip_sort`. Arrow keys are handled in an application event filter so they still work when the burst list has focus, and they are left alone when a slider, the burst-gap box, or the model dropdown has focus.

## Subject detection

The scoring panel lists four models. Choosing one turns subject scoring on. When Ultralytics is not importable, the window installs it before the scan. **Rescan** with that checkbox on calls `ensure_model` when that `.pt` file is missing or too small to be a weights archive, then scores the open folder with the saved file. The download is written to `filename.pt.part` and renamed only after the file is a large zip (PyTorch weights start with `PK`). Cancel deletes the partial file. `load_yolo_detector` imports Ultralytics only after that file is on disk and does not download anything. Detections are limited to the COCO animal and bird classes. If import fails, the file is missing, or a frame has no animal, the frame keeps the normal sharpness score.
