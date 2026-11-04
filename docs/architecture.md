# Outburst architecture

Outburst is a desktop app. The scoring pipeline does not import Qt, so it can be tested and reused without a window. The window is a client of that pipeline.

## Layout

| Module | Role |
| --- | --- |
| `scanner.py` | Finds RAW and JPEG files. A JPEG with the same stem as a RAW in the same folder is the preview pair, not a second frame. |
| `metadata.py` | Reads `DateTimeOriginal` and `SubSecTimeOriginal`. Uses exiftool when it is installed. Otherwise it uses exifread for JPEG and TIFF-based RAW, then a built-in TIFF/JPEG reader, then a date string in the file header. exifread does not recognize CR3, RAF, or X3F. |
| `grouping.py` | Sorts by capture time and groups consecutive frames whose gap is strictly below the threshold. Frames with no capture time stay alone. |
| `preview.py` | Loads a preview. Paired JPEGs win. Otherwise rawpy's embedded thumbnail is used. A byte scan for an embedded JPEG is the fallback. Full demosaic is never requested. |
| `scoring.py` | Laplacian variance, optional Tenengrad, exposure penalties, and the weighted composite. Pure functions. |
| `subject.py` | Optional YOLO detector and the model catalog. Device order is Apple Silicon MPS, then NVIDIA CUDA, then CPU. `ensure_model` downloads a missing weights file into the models folder. `load_yolo_detector` only opens a file that is already there. |
| `cache.py` | SQLite plus JPEG previews in the user cache directory. |
| `pipeline.py` | Ties the steps together and rebuilds scores or groups when weights or the gap change. |
| `output.py` | XMP sidecars, keeper copies or hard links, JSON and CSV reports. |
| `gui/` | PySide6 window, filmstrip, preview, and the background scan thread. |

## Pipeline

1. List the folder. Hidden files and hidden directories are skipped. Subfolders are included only when the checkbox is on.
2. Reuse a cache row when the path, size, modification time, score version, and subject-model key match and the preview files are still on disk.
3. For cache misses, read capture time, decode a preview, and shrink the long edge to 1024 px before measuring. A 360 px thumbnail is stored for the filmstrip.
4. If a subject detector is active, it runs on that same 1024 px image. The box is stored in those coordinates and drawn on the preview.
5. Group the frames and hand a `Library` to the window.

Changing the sharpness, exposure, or center sliders recomputes the composite from cached measurements. Changing the burst gap only regroups. Neither step rereads the RAW files. Turning subject detection on starts a new scan, because that measurement is not in the plain cache row.

## Scoring

Measurements:

- **Sharpness** is the variance of a Laplacian on the grayscale preview. Higher means more fine detail.
- **Center sharpness** uses the middle 60% of width and height.
- **Tenengrad** (mean squared Sobel gradient) is stored and shown as edge energy. It does not affect the rank.
- **Exposure** is 0 to 1. Pixels at or above 250 count as clipped highlights. Pixels at or below 5 count as crushed shadows. Highlights are penalized more heavily. A frame far from a mid tone is penalized a little.
- **Subject sharpness**, when a box exists, is the Laplacian of that box expanded by 12%.

The sharpness that enters the score is:

```text
center_mix * center_sharpness + (1 - center_mix) * sharpness
```

If subject scoring is on and a subject was found, that subject Laplacian replaces the blend.

It is then mapped to 0..1 with a fixed scale so one frame does not depend on the rest of the burst:

```text
normalized = sharpness / (sharpness + 300)
composite  = (sharpness_weight * normalized + exposure_weight * exposure)
             / (sharpness_weight + exposure_weight)
```

The window shows `composite * 100` as a whole number. The recommended frame is the highest composite. Ties prefer a higher full-frame Laplacian, then the earlier filename.

`SCORE_VERSION` in `scoring.py` is part of the cache key. Bump it when the measurement formula changes.

## Cache and decisions

The database lives in the user cache directory (`~/Library/Caches/Outburst` on macOS, `%LOCALAPPDATA%\Outburst\Cache` on Windows, `$XDG_CACHE_HOME/outburst` or `~/.cache/outburst` on Linux). Keeper and reject marks are rows in the same database, keyed by the photograph's path. They survive a restart and a cache clear. They are not written into the photograph.

`Cache.forget_folder` deletes measurement rows and their preview files for paths inside one folder. **Rescan** calls it before reading that folder again. `Cache.clear` does the same for every folder. Neither call deletes decision rows.

## What is allowed to be written

| Action | Where it writes |
| --- | --- |
| Scan and score | Cache directory only |
| Keep / reject | Cache directory only |
| Write XMP | A new or updated `<stem>.xmp` beside the photo, and only after confirmation |
| Save keepers | A folder the photographer chooses. Copy, or a hard link with a copy fallback. Only frames whose decision is keep |
| Export report | `outburst-report.json` and `outburst-report.csv` in a chosen folder |
| Download a subject model | The models folder: `models` in this project when Outburst is run from the checkout, `models` inside the cache directory otherwise, or `OUTBURST_MODELS` when that is set |

Photograph files are opened for reading. Export uses `shutil.copy2` or `os.link`. If an XMP sidecar exists and does not look like XMP, Outburst leaves it alone and reports the file.

## Window

`AnalyzeWorker` is a `QThread`. It emits progress and a finished `Library`. The window ignores results from a scan that was cancelled or replaced, using a scan id. `ModelFetchWorker` is a separate thread for the weights download. The burst list hides one-frame bursts until **Show single-frame bursts** is on. When a library is shown, each burst that has no keeper yet gets its recommended frame marked keep. **Save keepers** writes only those marks. The preview has **Mark as keeper** and **Reject** buttons besides the A and X keys. Arrow keys are handled in an application event filter so they still work when the burst list has focus, and they are left alone when a slider, the burst-gap box, or the model dropdown has focus.

## Subject detection

The scoring panel lists four models. Choosing one turns subject scoring on. **Rescan** with that checkbox on calls `ensure_model` when that `.pt` file is missing or too small to be a weights archive, then scores the open folder with the saved file. The download is written to `filename.pt.part` and renamed only after the file is a large zip (PyTorch weights start with `PK`). Cancel deletes the partial file. `load_yolo_detector` imports Ultralytics only after that file is on disk and does not download anything. Detections are limited to the COCO animal and bird classes. If import fails, the file is missing, or a frame has no animal, the frame keeps the normal sharpness score.
