"""Private Python used by a packaged Outburst install for subject and bird eye detection.

The Windows setup program is not a Python interpreter, so ``pip`` cannot install into Outburst.exe. The install button downloads the official embeddable Python, installs Ultralytics there, and scores previews by sending ``.npy`` files to ``outburst.detect_worker``. A normal ``python -m outburst`` install keeps using the running interpreter and does not call this module.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np

from outburst.subject import SUBJECT_PIP_SPEC, Detection

log = logging.getLogger(__name__)

EMBED_PYTHON_VERSION = "3.12.7"
EMBED_PYTHON_URL = (
    "https://www.python.org/ftp/python/"
    f"{EMBED_PYTHON_VERSION}/python-{EMBED_PYTHON_VERSION}-embed-amd64.zip"
)
EMBED_PYTHON_SHA256 = "0d57bb6cb078b74d23dbfe91f77d6780d45bed328911609f1f7ee2ba1606bf44"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
_STAMP_NAME = ".outburst-subject-ready"
_SESSION: DetectionSession | None = None


def detection_runtime_dir() -> Path:
    """
    Folder that holds the private Python for a packaged install.

    Inputs: none. OUTBURST_DETECTION_RUNTIME overrides the folder when it is set.
    Output: the directory, created if it was missing.
    Uses: the install button and the detection worker.
    Limitations: on Windows the default is under LocalAppData\\Outburst\\detection. Photographs are not written here.
    """

    override = os.environ.get("OUTBURST_DETECTION_RUNTIME")
    if override:
        path = Path(override).expanduser()
    elif sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else Path.home() / "AppData" / "Local"
        path = base / "Outburst" / "detection"
    else:
        path = Path.home() / ".local" / "share" / "Outburst" / "detection"
    path.mkdir(parents=True, exist_ok=True)
    return path


def private_runtime_ready() -> bool:
    """
    Report whether the private Python already has Ultralytics.

    Inputs: none.
    Output: True when python.exe and the ready stamp are both in the detection folder.
    Uses: the packaged app before it shows the detection controls.
    Limitations: this does not import Ultralytics in the frozen process. A deleted stamp reports not ready.
    """

    root = detection_runtime_dir()
    return _python_exe(root).is_file() and (root / _STAMP_NAME).is_file()


def install_private_runtime(
    should_cancel: object | None = None,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """
    Download a private Python and install Ultralytics into it.

    Inputs: should_cancel, an optional check polled while downloads and pip run. None never cancels. on_status, an optional callback that receives a short progress line.
    Output: no return value. Raises SubjectRuntimeCancelled when the check asks to stop, and RuntimeError when a download or pip fails.
    Uses: the install button on a packaged Windows app.
    Limitations: the Python build is the official 64-bit Windows embeddable package. PyTorch comes in with Ultralytics and can take several minutes. The status line names the step and counts bytes in the pip cache. Model weights are not downloaded here.
    """

    root = detection_runtime_dir()
    python = _python_exe(root)
    if not python.is_file():
        _report(on_status, "Downloading Python for subject and bird eye detection")
        _download_embeddable(root, should_cancel, on_status)
    _enable_site(root)
    if not _command_ok([str(python), "-m", "pip", "--version"], should_cancel):
        _report(on_status, "Installing pip")
        _install_pip(python, root, should_cancel, on_status)
    _report(
        on_status,
        "Installing Ultralytics and PyTorch. This download is large and can take several minutes.",
    )
    cache_dir = root / "pip-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    code, _stdout, stderr = _run_logged(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            SUBJECT_PIP_SPEC,
        ],
        root / "install.log",
        should_cancel,
        on_status=on_status,
        env_extra={"PIP_CACHE_DIR": str(cache_dir)},
    )
    if code != 0:
        detail = (stderr or "pip install failed").strip()
        raise RuntimeError(detail[-800:])
    _report(on_status, "Checking that Ultralytics imports. This can take a minute.")
    code, _stdout, stderr = _run(
        [str(python), "-c", "import ultralytics"],
        should_cancel,
    )
    if code != 0:
        detail = (stderr or "Ultralytics did not import in the private Python.").strip()
        raise RuntimeError(detail[-800:])
    (root / _STAMP_NAME).write_text("ultralytics\n", encoding="utf-8")


def detection_source_dir() -> Path:
    """
    Folder that contains the outburst package for the private Python.

    Inputs: none.
    Output: the src directory in a checkout, or the detection_src folder shipped inside a frozen app.
    Uses: the command that starts detect_worker.
    Limitations: a frozen app reads the copy PyInstaller stored. This does not install that package with pip.
    """

    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            return Path(meipass) / "detection_src"
    return Path(__file__).resolve().parents[1]


class DetectionSession:
    """One private Python process that keeps the detection models loaded."""

    def __init__(
        self,
        python: Path | None = None,
        source: Path | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        """
        Remember which Python will score previews.

        Inputs: python, the private interpreter. The detection folder is used when python is omitted. source, the folder on PYTHONPATH. extra_env, variables added for the worker, such as the stub flag.
        Output: no return value. The process starts on the first request.
        Uses: SidecarSubjectDetector and SidecarEyeFinder.
        Limitations: constructing the session does not download Python or start a process.
        """

        self._python = python if python is not None else _python_exe(detection_runtime_dir())
        self._source = source if source is not None else detection_source_dir()
        self._extra_env = dict(extra_env or {})
        self._process: subprocess.Popen[str] | None = None
        self._stderr: list[str] = []
        self._loaded_subject: str | None = None
        self._loaded_eye: str | None = None
        self._lock = threading.Lock()

    def ensure_subject(self, model_path: Path) -> None:
        """
        Load the subject weights in the worker if this path is not already loaded.

        Inputs: model_path, the .pt file.
        Output: no return value. A load command is sent when the path changed.
        Uses: SidecarSubjectDetector before a batch.
        Limitations: a worker that exits is started again and the model is loaded again.
        """

        path = str(model_path)
        with self._lock:
            self._ensure_started()
            if self._loaded_subject == path:
                return
            self._request({"cmd": "load", "subject": path})
            self._loaded_subject = path

    def ensure_eye(self, model_path: Path) -> None:
        """
        Load the bird eye weights in the worker if this path is not already loaded.

        Inputs: model_path, the .pt file.
        Output: no return value. A load command is sent when the path changed.
        Uses: SidecarEyeFinder before a batch.
        Limitations: a worker that exits is started again and the model is loaded again.
        """

        path = str(model_path)
        with self._lock:
            self._ensure_started()
            if self._loaded_eye == path:
                return
            self._request({"cmd": "load", "eye": path})
            self._loaded_eye = path

    def subject_boxes(self, files: list[str]) -> list[dict[str, object] | None]:
        """
        Ask the worker to score subject boxes for saved previews.

        Inputs: files, ``.npy`` paths in batch order.
        Output: one box dict or None per file.
        Uses: SidecarSubjectDetector.detect_many.
        Limitations: the caller holds no lock of its own. This call takes the session lock.
        """

        with self._lock:
            self._ensure_started()
            reply = self._request({"cmd": "subject", "files": files})
        boxes = reply.get("boxes")
        if not isinstance(boxes, list):
            raise RuntimeError("Subject detection returned no boxes.")
        return boxes

    def eyes(self, files: list[str]) -> list[dict[str, object] | None]:
        """
        Ask the worker to score eyes for saved previews.

        Inputs: files, ``.npy`` paths in batch order.
        Output: one eye dict or None per file.
        Uses: SidecarEyeFinder.find_many.
        Limitations: the caller holds no lock of its own. This call takes the session lock.
        """

        with self._lock:
            self._ensure_started()
            reply = self._request({"cmd": "eye", "files": files})
        eyes = reply.get("eyes")
        if not isinstance(eyes, list):
            raise RuntimeError("Bird eye detection returned no eyes.")
        return eyes

    def close(self) -> None:
        """
        Stop the worker process.

        Inputs: none.
        Output: no return value. A running worker is asked to quit, then terminated if it is still running.
        Uses: closing the app, and tests.
        Limitations: a worker that ignores quit is killed. Loaded model paths are forgotten.
        """

        with self._lock:
            process = self._process
            self._process = None
            self._loaded_subject = None
            self._loaded_eye = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin is not None:
                process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                process.stdin.flush()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def _ensure_started(self) -> None:
        """
        Start the worker when it is not running.

        Inputs: none.
        Output: no return value. A dead process is replaced and its loaded paths are cleared.
        Uses: every request.
        Limitations: the caller must hold the session lock. This does not load a model.
        """

        if self._process is not None and self._process.poll() is None:
            return
        self._loaded_subject = None
        self._loaded_eye = None
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self._source)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env.update(self._extra_env)
        self._stderr = []
        process = subprocess.Popen(
            [str(self._python), "-u", "-m", "outburst.detect_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            **_hidden_windows_kwargs(),
        )
        self._process = process
        thread = threading.Thread(target=_drain, args=(process, self._stderr), daemon=True)
        thread.start()

    def _request(self, payload: dict[str, object]) -> dict[str, object]:
        """
        Send one command and read one reply.

        Inputs: payload, the JSON command.
        Output: the decoded reply. A failed reply raises RuntimeError.
        Uses: load, subject, and eye commands.
        Limitations: the caller must hold the session lock. stdout must stay one JSON object per line.
        """

        process = self._process
        if process is None or process.stdin is None or process.stdout is None:
            raise RuntimeError("Subject detection is not running.")
        process.stdin.write(json.dumps(payload) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError(self._failure_text())
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(str(reply.get("error") or "Detection failed."))
        return reply

    def _failure_text(self) -> str:
        """
        Describe a worker that closed without a reply.

        Inputs: none.
        Output: the last stderr lines, or a short sentence when there are none.
        Uses: _request.
        Limitations: only the lines already drained are included.
        """

        detail = " ".join(self._stderr[-8:]).strip()
        if detail:
            return detail[-800:]
        return "Subject detection stopped before it returned a result."


class SidecarSubjectDetector:
    """Subject detector that asks the private Python for boxes."""

    def __init__(self, model_path: Path, session: DetectionSession | None = None) -> None:
        """
        Keep the weights path and the shared worker.

        Inputs: model_path, the .pt file; session, an optional worker. The app-wide session is used when session is omitted.
        Output: no return value.
        Uses: load_yolo_detector on a packaged Windows app.
        Limitations: constructing this does not load the weights. The file is checked by the caller.
        """

        self._model_path = model_path
        self._session = session if session is not None else default_session()

    def detect(self, image: np.ndarray) -> Detection | None:
        """
        Score one preview.

        Inputs: image, a BGR preview.
        Output: the Detection, or None.
        Uses: callers that do not batch.
        Limitations: this calls detect_many with one image.
        """

        found = self.detect_many([image])
        if not found:
            return None
        return found[0]

    def detect_many(self, images: list[np.ndarray]) -> list[Detection | None]:
        """
        Score a batch in the private Python.

        Inputs: images, BGR previews in order.
        Output: one Detection or None per image.
        Uses: the scan pipeline.
        Limitations: each preview is written to a temporary ``.npy`` file and removed afterward. The model stays in the worker.
        """

        if not images:
            return []
        self._session.ensure_subject(self._model_path)
        with tempfile.TemporaryDirectory(prefix="outburst-detect-") as folder:
            files = _save_previews(Path(folder), images)
            payloads = self._session.subject_boxes(files)
        found: list[Detection | None] = []
        for item in payloads:
            found.append(_detection_from_payload(item))
        if len(found) < len(images):
            found.extend([None] * (len(images) - len(found)))
        return found[: len(images)]


class SidecarEyeFinder:
    """Bird eye finder that asks the private Python for eyes."""

    def __init__(self, model_path: Path, session: DetectionSession | None = None) -> None:
        """
        Keep the weights path and the shared worker.

        Inputs: model_path, the .pt file; session, an optional worker. The app-wide session is used when session is omitted.
        Output: no return value.
        Uses: load_bird_eye_detector on a packaged Windows app.
        Limitations: constructing this does not load the weights.
        """

        self._model_path = model_path
        self._session = session if session is not None else default_session()

    def find_many(self, images: list[np.ndarray]) -> list[object]:
        """
        Score a batch of eyes in the private Python.

        Inputs: images, BGR previews in order.
        Output: one Eye or None per image.
        Uses: the scan pipeline.
        Limitations: each preview is written to a temporary ``.npy`` file. Sharpness is measured in the private Python.
        """

        from outburst.eyes import Eye

        if not images:
            return []
        self._session.ensure_eye(self._model_path)
        with tempfile.TemporaryDirectory(prefix="outburst-eye-") as folder:
            files = _save_previews(Path(folder), images)
            payloads = self._session.eyes(files)
        found: list[Eye | None] = []
        for item in payloads:
            found.append(_eye_from_payload(item))
        if len(found) < len(images):
            found.extend([None] * (len(images) - len(found)))
        return found[: len(images)]


def default_session() -> DetectionSession:
    """
    Return the worker kept for this app process.

    Inputs: none.
    Output: the shared DetectionSession. The first call creates it and registers it to close when the app exits.
    Uses: the packaged detectors.
    Limitations: one process is shared by subject and bird eye detection. Tests that need a stub pass their own session.
    """

    global _SESSION
    if _SESSION is None:
        _SESSION = DetectionSession()
        atexit.register(_SESSION.close)
    return _SESSION


def _python_exe(root: Path) -> Path:
    """
    Path of the private interpreter.

    Inputs: root, the detection folder.
    Output: python.exe inside the python subfolder on Windows, or python on other systems.
    Uses: the install and the ready check.
    Limitations: the file is not created here.
    """

    name = "python.exe" if sys.platform == "win32" else "python"
    return root / "python" / name


def _download_embeddable(
    root: Path,
    should_cancel: object | None,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """
    Download and unpack the embeddable Python package.

    Inputs: root, the detection folder; should_cancel, the cancel check; on_status, an optional progress line.
    Output: no return value. python.exe is extracted under root/python.
    Uses: install_private_runtime the first time.
    Limitations: the zip hash must match EMBED_PYTHON_SHA256. An unsafe path inside the zip raises RuntimeError. A cancel removes the partial zip.
    """

    from outburst.subject import SubjectRuntimeCancelled

    destination = root / "python-embed.zip"
    try:
        _download(
            EMBED_PYTHON_URL,
            destination,
            should_cancel,
            on_status,
            "Downloading Python",
        )
    except SubjectRuntimeCancelled:
        destination.unlink(missing_ok=True)
        raise
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if digest != EMBED_PYTHON_SHA256:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "The Python download did not match the expected file. "
            "Subject and bird eye detection were not installed."
        )
    python_dir = root / "python"
    if python_dir.exists():
        shutil.rmtree(python_dir)
    python_dir.mkdir(parents=True)
    with zipfile.ZipFile(destination) as archive:
        for info in archive.infolist():
            relative = Path(info.filename.replace("\\", "/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError("The Python download contained an unsafe path.")
            target = python_dir / relative
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle)
    destination.unlink(missing_ok=True)


def _enable_site(root: Path) -> None:
    """
    Let the embeddable Python use site-packages.

    Inputs: root, the detection folder.
    Output: no return value. The ``._pth`` file lists Lib\\site-packages and imports site.
    Uses: install_private_runtime after the zip is unpacked.
    Limitations: a Python folder with no ``._pth`` file raises RuntimeError. The standard library zip name is kept.
    """

    python_dir = root / "python"
    matches = list(python_dir.glob("python*._pth"))
    if not matches:
        raise RuntimeError("The private Python is missing its path file.")
    path_file = matches[0]
    zip_name = path_file.name.replace("._pth", ".zip")
    path_file.write_text(
        f"{zip_name}\n.\nLib\\site-packages\nimport site\n",
        encoding="utf-8",
    )


def _install_pip(
    python: Path,
    root: Path,
    should_cancel: object | None,
    on_status: Callable[[str], None] | None = None,
) -> None:
    """
    Bootstrap pip into the private Python.

    Inputs: python, the interpreter; root, the detection folder; should_cancel, the cancel check; on_status, an optional progress line.
    Output: no return value. get-pip.py is downloaded and run.
    Uses: install_private_runtime when pip is missing.
    Limitations: a failed bootstrap raises RuntimeError. The script is removed after it runs. Output is read while the script runs so a long log cannot stall the installer.
    """

    script = root / "get-pip.py"
    _download(GET_PIP_URL, script, should_cancel, on_status, "Downloading pip")
    code, _stdout, stderr = _run([str(python), str(script), "--no-warn-script-location"], should_cancel)
    script.unlink(missing_ok=True)
    if code != 0:
        detail = (stderr or "pip could not be installed.").strip()
        raise RuntimeError(detail[-800:])


def _download(
    url: str,
    destination: Path,
    should_cancel: object | None,
    on_status: Callable[[str], None] | None = None,
    label: str = "Downloading",
) -> None:
    """
    Save a URL to a file and stop when the cancel check asks.

    Inputs: url, the address; destination, the file to write; should_cancel, the cancel check; on_status, an optional progress line; label, the text before the received size.
    Output: no return value. The file is replaced. on_status is called about once per megabyte.
    Uses: the Python zip and get-pip.py.
    Limitations: a cancel leaves a partial file for the caller to remove. The request uses a 60 second timeout and the certifi certificate bundle.
    """

    from outburst.subject import SubjectRuntimeCancelled, _https_context

    request = urllib.request.Request(url, headers={"User-Agent": "Outburst"})
    with urllib.request.urlopen(
        request, timeout=60, context=_https_context()
    ) as response, destination.open("wb") as handle:
        downloaded = 0
        reported = -1
        while True:
            if callable(should_cancel) and should_cancel():
                raise SubjectRuntimeCancelled()
            chunk = response.read(256 * 1024)
            if not chunk:
                return
            handle.write(chunk)
            downloaded += len(chunk)
            megabytes = downloaded // (1024 * 1024)
            if on_status is not None and megabytes != reported:
                reported = megabytes
                _report(on_status, f"{label}, {megabytes} MB")


def _command_ok(args: list[str], should_cancel: object | None) -> bool:
    """
    Run a short command and report whether it exited 0.

    Inputs: args, the command; should_cancel, the cancel check.
    Output: True when the exit code is 0. A cancel is re-raised.
    Uses: the pip presence check.
    Limitations: output is discarded except for the exit code. A command that cannot start returns False.
    """

    from outburst.subject import SubjectRuntimeCancelled

    try:
        code, _stdout, _stderr = _run(args, should_cancel)
    except SubjectRuntimeCancelled:
        raise
    except OSError:
        return False
    return code == 0


def _run(args: list[str], should_cancel: object | None) -> tuple[int, str, str]:
    """
    Run a command and capture its output.

    Inputs: args, the command; should_cancel, the cancel check.
    Output: exit code, stdout, and stderr. A cancel terminates the process and raises SubjectRuntimeCancelled.
    Uses: pip checks, get-pip, and the import check.
    Limitations: stdout and stderr are read while the process runs, so a long log cannot fill the pipe and stall. Long installs still use _run_logged so the status line can follow the download.
    """

    process = _spawn(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    threads, stdout_parts, stderr_parts = _start_drains(process)
    try:
        _wait_process(process, should_cancel)
    finally:
        for thread in threads:
            thread.join(timeout=2)
    return process.returncode or 0, "".join(stdout_parts), "".join(stderr_parts)


def _run_logged(
    args: list[str],
    log_path: Path,
    should_cancel: object | None,
    on_status: Callable[[str], None] | None = None,
    env_extra: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """
    Run a long command and append its output to a log file.

    Inputs: args, the command; log_path, the file that receives stdout and stderr; should_cancel, the cancel check; on_status, an optional progress line; env_extra, variables added for this process. PIP_CACHE_DIR is measured for the received-byte count.
    Output: exit code, an empty stdout, and the last part of the log. A cancel raises SubjectRuntimeCancelled.
    Uses: pip install of Ultralytics.
    Limitations: the log grows across attempts. Only the tail is returned. The received count is the growth of PIP_CACHE_DIR during this run. Stdin is discarded so a windowed parent cannot leave pip waiting for a keypress.
    """

    log_path.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(env_extra["PIP_CACHE_DIR"]) if env_extra and env_extra.get("PIP_CACHE_DIR") else None
    baseline = _tree_size(cache) if cache is not None else 0
    with log_path.open("a", encoding="utf-8", errors="replace") as handle:
        handle.write("\n--- installing ultralytics ---\n")
        handle.flush()
        process = _spawn(
            args,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env_extra=env_extra,
        )
        _wait_process(
            process,
            should_cancel,
            on_tick=lambda: _report_install(on_status, log_path, cache, baseline),
        )
        process.wait(timeout=10)
    tail = ""
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-800:]
    except OSError:
        tail = ""
    return process.returncode or 0, "", tail


def _spawn(
    args: list[str],
    *,
    stdout: object,
    stderr: object,
    env_extra: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """
    Start a hidden command with stdin closed.

    Inputs: args, the command; stdout and stderr, the pipes or files that receive output; env_extra, variables added to the current environment.
    Output: the running process.
    Uses: _run and _run_logged.
    Limitations: PYTHONPATH and PYTHONHOME are removed so a packaged app does not point the private Python back into Outburst. The window stays hidden on Windows. Stdin is discarded.
    """

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=stdout,  # type: ignore[arg-type]
        stderr=stderr,  # type: ignore[arg-type]
        text=True,
        env=env,
        **_hidden_windows_kwargs(),
    )


def _start_drains(
    process: subprocess.Popen[str],
) -> tuple[list[threading.Thread], list[str], list[str]]:
    """
    Read both pipes on background threads.

    Inputs: process, a command started with stdout and stderr pipes.
    Output: the reader threads, the stdout pieces, and the stderr pieces.
    Uses: _run while it waits.
    Limitations: a missing pipe is skipped. The threads stop when the pipe closes.
    """

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    threads: list[threading.Thread] = []
    for stream, parts in (
        (getattr(process, "stdout", None), stdout_parts),
        (getattr(process, "stderr", None), stderr_parts),
    ):
        if stream is None:
            continue
        thread = threading.Thread(target=_read_stream, args=(stream, parts), daemon=True)
        thread.start()
        threads.append(thread)
    return threads, stdout_parts, stderr_parts


def _read_stream(stream: object, parts: list[str]) -> None:
    """
    Copy one stream into a list until it closes.

    Inputs: stream, an iterable of text lines; parts, the list that stores them.
    Output: no return value. The list grows by one entry per line.
    Uses: _start_drains.
    Limitations: an unreadable stream stops the copy. This runs on a daemon thread.
    """

    try:
        for line in stream:  # type: ignore[attr-defined]
            parts.append(str(line))
    except Exception:
        return


def _wait_process(
    process: subprocess.Popen[str],
    should_cancel: object | None,
    on_tick: Callable[[], None] | None = None,
) -> None:
    """
    Wait until a command exits, and stop it when the cancel check asks.

    Inputs: process, the command; should_cancel, the cancel check; on_tick, an optional callback polled about every second.
    Output: no return value. SubjectRuntimeCancelled is raised after the process is stopped.
    Uses: _run and _run_logged.
    Limitations: the callback is skipped when it is omitted. A process that ignores terminate is killed.
    """

    from outburst.subject import SubjectRuntimeCancelled

    last_tick = 0.0
    while process.poll() is None:
        if callable(should_cancel) and should_cancel():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            raise SubjectRuntimeCancelled()
        now = time.monotonic()
        if on_tick is not None and now - last_tick >= 0.8:
            last_tick = now
            on_tick()
        try:
            process.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            continue
        break


def _report(on_status: Callable[[str], None] | None, message: str) -> None:
    """
    Send one progress line when a callback was provided.

    Inputs: on_status, the callback or None; message, the status text.
    Output: no return value. A missing callback does nothing.
    Uses: the private Python install.
    Limitations: the message is not written to the log.
    """

    if on_status is not None:
        on_status(message)


def _report_install(
    on_status: Callable[[str], None] | None,
    log_path: Path,
    cache: Path | None,
    baseline: int,
) -> None:
    """
    Describe the Ultralytics install from the log and the pip cache.

    Inputs: on_status, the callback or None; log_path, the install log; cache, the pip cache or None; baseline, the cache size in bytes when pip started.
    Output: no return value. The callback receives the latest log line and how many megabytes arrived after baseline.
    Uses: _run_logged about once a second.
    Limitations: a cache outside PIP_CACHE_DIR is not counted. The log line is shortened to 160 characters.
    """

    if on_status is None:
        return
    received = 0
    if cache is not None:
        received = max(0, _tree_size(cache) - baseline)
    megabytes = received / (1024 * 1024)
    line = _last_log_line(log_path)
    amount = f"{megabytes:.0f} MB received"
    if line:
        on_status(f"{line} — {amount}")
        return
    on_status(f"Installing Ultralytics and PyTorch — {amount}")


def _last_log_line(path: Path) -> str:
    """
    Read the latest useful line from an install log.

    Inputs: path, the log file.
    Output: the last non-empty line, with progress-bar returns treated as line breaks, or an empty string.
    Uses: _report_install.
    Limitations: only the last 2000 bytes are read. Marker lines that start with --- are skipped. A missing file returns an empty string.
    """

    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 2000))
            data = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    for line in reversed(data.replace("\r", "\n").splitlines()):
        text = line.strip()
        if text and not text.startswith("---"):
            return text[:160]
    return ""


def _tree_size(path: Path | None) -> int:
    """
    Add up the file sizes under a directory.

    Inputs: path, the directory, or None.
    Output: the total size in bytes. A missing directory is 0.
    Uses: the received-byte count during pip.
    Limitations: directories are not followed through symlinks. A file that disappears during the walk is skipped.
    """

    if path is None or not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def _save_previews(folder: Path, images: list[np.ndarray]) -> list[str]:
    """
    Write BGR previews as ``.npy`` files.

    Inputs: folder, an existing directory; images, the previews in order.
    Output: the file paths as strings.
    Uses: the sidecar detectors.
    Limitations: the arrays are copied into contiguous arrays. The caller deletes the folder.
    """

    paths: list[str] = []
    for index, image in enumerate(images):
        path = folder / f"{index}.npy"
        np.save(path, np.ascontiguousarray(image))
        paths.append(str(path))
    return paths


def _detection_from_payload(item: object) -> Detection | None:
    """
    Build a Detection from one worker box.

    Inputs: item, a dict with box, label, and confidence, or None.
    Output: a Detection, or None.
    Uses: SidecarSubjectDetector.
    Limitations: a payload that is not a dict with a four-number box returns None.
    """

    if not isinstance(item, dict):
        return None
    raw_box = item.get("box")
    if not isinstance(raw_box, list) or len(raw_box) < 4:
        return None
    box = tuple(int(value) for value in raw_box[:4])
    return Detection(box, str(item.get("label", "")), float(item.get("confidence", 0.0)))  # type: ignore[arg-type]


def _eye_from_payload(item: object) -> object:
    """
    Build an Eye from one worker eye.

    Inputs: item, a dict with x, y, radius, and sharpness, or None.
    Output: an Eye, or None.
    Uses: SidecarEyeFinder.
    Limitations: a payload that is not a dict returns None.
    """

    from outburst.eyes import Eye

    if not isinstance(item, dict):
        return None
    sharpness = item.get("sharpness")
    return Eye(
        int(item.get("x", 0)),
        int(item.get("y", 0)),
        int(item.get("radius", 0)),
        None if sharpness is None else float(sharpness),
    )


def _drain(process: subprocess.Popen[str], lines: list[str]) -> None:
    """
    Copy worker stderr into a short list so the pipe cannot fill.

    Inputs: process, the worker; lines, the list that stores the text.
    Output: no return value. The list keeps the last 40 lines.
    Uses: DetectionSession when the worker starts.
    Limitations: this runs on a daemon thread. It stops when stderr closes.
    """

    stderr = process.stderr
    if stderr is None:
        return
    for line in stderr:
        lines.append(line.rstrip())
        del lines[:-40]


def _hidden_windows_kwargs() -> dict[str, object]:
    """
    Stop Windows from opening a console for the private Python.

    Inputs: none.
    Output: subprocess keyword arguments that hide the window on Windows, or an empty dict on other platforms.
    Uses: pip and the detection worker.
    Limitations: the flags apply only on Windows.
    """

    if sys.platform != "win32" or not hasattr(subprocess, "STARTUPINFO"):
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "startupinfo": startupinfo,
        "creationflags": subprocess.CREATE_NO_WINDOW,
    }
