"""Map libraries used by the packaged app to Debian package names.

``packages_from_ldd`` reads one ``ldd`` transcript. ``packages_for_tree`` reads the NEEDED entries of the app's libraries with ``readelf`` and adds the libraries named by the X11 and Wayland platform plugins. The build script prints the Depends line. The build machine does not have to load Qt.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Alternatives cover the Debian t64 rename and the older package name.
SONAME_PACKAGES: dict[str, str] = {
    "ld-linux-aarch64.so.1": "libc6",
    "ld-linux-x86-64.so.2": "libc6",
    "libc.so.6": "libc6",
    "libdl.so.2": "libc6",
    "libm.so.6": "libc6",
    "libpthread.so.0": "libc6",
    "libresolv.so.2": "libc6",
    "librt.so.1": "libc6",
    "libutil.so.1": "libc6",
    "libgcc_s.so.1": "libgcc-s1",
    "libstdc++.so.6": "libstdc++6",
    "libgomp.so.1": "libgomp1",
    "libatomic.so.1": "libatomic1",
    "libz.so.1": "zlib1g",
    "libglib-2.0.so.0": "libglib2.0-0t64 | libglib2.0-0",
    "libgthread-2.0.so.0": "libglib2.0-0t64 | libglib2.0-0",
    "libgobject-2.0.so.0": "libglib2.0-0t64 | libglib2.0-0",
    "libgio-2.0.so.0": "libglib2.0-0t64 | libglib2.0-0",
    "libgmodule-2.0.so.0": "libglib2.0-0t64 | libglib2.0-0",
    "libpcre2-8.so.0": "libpcre2-8-0",
    "libffi.so.8": "libffi8",
    "libffi.so.7": "libffi7",
    "libGL.so.1": "libgl1",
    "libGLX.so.0": "libglx0",
    "libGLdispatch.so.0": "libglvnd0",
    "libOpenGL.so.0": "libopengl0",
    "libEGL.so.1": "libegl1",
    "libX11.so.6": "libx11-6",
    "libX11-xcb.so.1": "libx11-xcb1",
    "libXext.so.6": "libxext6",
    "libXrender.so.1": "libxrender1",
    "libXi.so.6": "libxi6",
    "libXfixes.so.3": "libxfixes3",
    "libXrandr.so.2": "libxrandr2",
    "libXcursor.so.1": "libxcursor1",
    "libXinerama.so.1": "libxinerama1",
    "libSM.so.6": "libsm6",
    "libICE.so.6": "libice6",
    "libxcb.so.1": "libxcb1",
    "libxcb-xkb.so.1": "libxcb-xkb1",
    "libxcb-cursor.so.0": "libxcb-cursor0",
    "libxcb-glx.so.0": "libxcb-glx0",
    "libxcb-icccm.so.4": "libxcb-icccm4",
    "libxcb-image.so.0": "libxcb-image0",
    "libxcb-keysyms.so.1": "libxcb-keysyms1",
    "libxcb-randr.so.0": "libxcb-randr0",
    "libxcb-render.so.0": "libxcb-render0",
    "libxcb-render-util.so.0": "libxcb-render-util0",
    "libxcb-shape.so.0": "libxcb-shape0",
    "libxcb-shm.so.0": "libxcb-shm0",
    "libxcb-sync.so.1": "libxcb-sync1",
    "libxcb-xfixes.so.0": "libxcb-xfixes0",
    "libxcb-util.so.1": "libxcb-util1",
    "libxkbcommon.so.0": "libxkbcommon0",
    "libxkbcommon-x11.so.0": "libxkbcommon-x11-0",
    "libfontconfig.so.1": "libfontconfig1",
    "libfreetype.so.6": "libfreetype6",
    "libexpat.so.1": "libexpat1",
    "libpng16.so.16": "libpng16-16t64 | libpng16-16",
    "libuuid.so.1": "libuuid1",
    "libdbus-1.so.3": "libdbus-1-3",
    "libnss3.so": "libnss3",
    "libnspr4.so": "libnspr4",
    "libssl.so.3": "libssl3t64 | libssl3",
    "libcrypto.so.3": "libssl3t64 | libssl3",
    "libasound.so.2": "libasound2t64 | libasound2",
    "libwayland-client.so.0": "libwayland-client0",
    "libwayland-cursor.so.0": "libwayland-cursor0",
    "libwayland-egl.so.1": "libwayland-egl1",
    "libxshmfence.so.1": "libxshmfence1",
    "libbsd.so.0": "libbsd0",
    "libmd.so.0": "libmd0",
    "libcap.so.2": "libcap2",
    "libdrm.so.2": "libdrm2",
    "libudev.so.1": "libudev1",
    "libb2.so.1": "libb2-1",
    "libbz2.so.1": "libbz2-1.0",
    "libgpg-error.so.0": "libgpg-error0",
    "libgcrypt.so.20": "libgcrypt20",
    "liblz4.so.1": "liblz4-1",
    "libsystemd.so.0": "libsystemd0",
    "libbrotlicommon.so.1": "libbrotli1",
    "libbrotlidec.so.1": "libbrotli1",
    "libXau.so.6": "libxau6",
    "libXdmcp.so.6": "libxdmcp6",
    "liblzma.so.5": "liblzma5",
    "libzstd.so.1": "libzstd1",
    "libpcre.so.3": "libpcre3",
    "libselinux.so.1": "libselinux1",
    "libmount.so.1": "libmount1",
    "libblkid.so.1": "libblkid1",
    "libgssapi_krb5.so.2": "libgssapi-krb5-2",
    "libkrb5.so.3": "libkrb5-3",
    "libk5crypto.so.3": "libkrb5-3",
    "libcom_err.so.2": "libcom-err2",
    "libkeyutils.so.1": "libkeyutils1",
    "libkrb5support.so.0": "libkrb5-3",
    "libcrypt.so.1": "libcrypt1",
    "libnsl.so.1": "libnsl2",
    "libtirpc.so.3": "libtirpc3t64 | libtirpc3",
    "libxcrypt.so.1": "libcrypt1",
}

EXIFTOOL_PACKAGE = "libimage-exiftool-perl"

# Libraries the app loads, plus the plugins that open the window on X11 or Wayland.
# The GTK theme plugin stays optional, so a system without GTK can still open the window.
_APP_LIBRARIES = (
    "bin/python3.12",
    "lib/python3.12/site-packages/PySide6/QtCore.abi3.so",
    "lib/python3.12/site-packages/PySide6/QtGui.abi3.so",
    "lib/python3.12/site-packages/PySide6/QtWidgets.abi3.so",
    "lib/python3.12/site-packages/PySide6/libpyside6.abi3.so*",
    "lib/python3.12/site-packages/shiboken6/*.so*",
    "lib/python3.12/site-packages/cv2/cv2*.so",
    "lib/python3.12/site-packages/numpy/_core/_multiarray_umath*.so",
    "lib/python3.12/site-packages/numpy/linalg/_umath_linalg*.so",
    "lib/python3.12/site-packages/numpy/fft/_pocketfft_umath*.so",
    "lib/python3.12/site-packages/rawpy/_rawpy*.so",
    "lib/python3.12/site-packages/PySide6/Qt/plugins/platforms/libqxcb.so",
    "lib/python3.12/site-packages/PySide6/Qt/plugins/platforms/libqwayland.so",
    "lib/python3.12/site-packages/PySide6/Qt/plugins/xcbglintegrations/libqxcb-egl-integration.so",
    "lib/python3.12/site-packages/PySide6/Qt/plugins/xcbglintegrations/libqxcb-glx-integration.so",
)


def package_for_filename(name: str) -> str | None:
    """
    Find the Debian package for one loaded file name.

    Inputs: name, the file name from a process map. It may be a versioned file such as libz.so.1.3.
    Output: the package name, or None when the file is not a mapped library.
    Uses: packages_from_paths.
    Limitations: the longest matching soname wins. A file whose name does not contain .so is ignored by the caller.
    """

    if name in SONAME_PACKAGES:
        return SONAME_PACKAGES[name]
    best: str | None = None
    best_key = ""
    for soname, package in SONAME_PACKAGES.items():
        if name.startswith(soname + ".") and len(soname) > len(best_key):
            best = package
            best_key = soname
    return best


def packages_from_ldd(text: str, prefix: str) -> list[str]:
    """
    Turn an ldd transcript into Debian package names.

    Inputs: text, the combined ldd output; prefix, the packaged Python directory. A library resolved inside prefix is skipped.
    Output: package names, in first-seen order, without duplicates.
    Uses: the Debian build before it writes the control file.
    Limitations: a missing library or an unmapped soname raises RuntimeError. linux-vdso is ignored. This does not install packages.
    """

    root = str(Path(prefix).resolve())
    found: list[str] = []
    seen: set[str] = set()
    problems: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "not a dynamic executable" in line or "statically linked" in line:
            continue
        if "=>" not in line:
            name = line.split()[0]
            if name.startswith("linux-vdso") or name.startswith("linux-gate"):
                continue
            try:
                direct = str(Path(name).resolve())
            except OSError:
                direct = name
            if direct == root or direct.startswith(root + "/"):
                continue
            loader = Path(name).name
            if loader.startswith("ld-linux") or loader.startswith("ld-musl"):
                package = SONAME_PACKAGES.get(loader, "libc6")
                if package not in seen:
                    seen.add(package)
                    found.append(package)
                continue
            problems.append(f"Unrecognized ldd line: {line}")
            continue
        soname, _arrow, rest = line.partition(" => ")
        soname = soname.strip()
        resolved = rest.split()[0] if rest else ""
        if resolved == "not":
            problems.append(f"Missing system library: {soname}")
            continue
        try:
            resolved_path = str(Path(resolved).resolve())
        except OSError:
            resolved_path = resolved
        if resolved_path == root or resolved_path.startswith(root + "/"):
            continue
        package = SONAME_PACKAGES.get(soname)
        if package is None:
            problems.append(f"No Debian package is mapped for {soname} ({resolved})")
            continue
        if package not in seen:
            seen.add(package)
            found.append(package)
    if problems:
        unique = list(dict.fromkeys(problems))
        raise RuntimeError("\n".join(unique))
    return found


def packages_from_paths(paths: list[str], prefix: str) -> list[str]:
    """
    Turn loaded library paths into Debian package names.

    Inputs: paths, absolute files opened by the process; prefix, the packaged Python directory. A file inside prefix is skipped.
    Output: package names, in first-seen order, without duplicates.
    Uses: packages_for_tree after the app imports its libraries.
    Limitations: an unmapped file outside prefix raises RuntimeError. This does not install packages.
    """

    root = str(Path(prefix).resolve())
    found: list[str] = []
    seen: set[str] = set()
    problems: list[str] = []
    for raw in paths:
        try:
            resolved = str(Path(raw).resolve())
        except OSError:
            resolved = raw
        if resolved == root or resolved.startswith(root + "/"):
            continue
        soname = Path(resolved).name
        if ".so" not in soname:
            continue
        if soname.startswith("ld-linux") or soname.startswith("ld-musl"):
            package = package_for_filename(soname) or "libc6"
        else:
            package = package_for_filename(soname)
        if package is None:
            problems.append(f"No Debian package is mapped for {soname} ({resolved})")
            continue
        if package not in seen:
            seen.add(package)
            found.append(package)
    if problems:
        raise RuntimeError("\n".join(dict.fromkeys(problems)))
    return found


def packages_from_sonames(sonames: list[str]) -> list[str]:
    """
    Turn plugin NEEDED entries into Debian package names.

    Inputs: sonames, names from readelf, including Qt libraries that ship inside the package.
    Output: package names, in first-seen order, without duplicates. Qt, shiboken, and libpython names are skipped.
    Uses: the Debian package tests.
    Limitations: Qt, shiboken, and libpython names are skipped. An unmapped system soname raises RuntimeError. This does not install packages.
    """

    found: list[str] = []
    seen: set[str] = set()
    problems: list[str] = []
    for soname in sonames:
        if soname.startswith(("libQt", "libshiboken", "libpython")):
            continue
        package = package_for_filename(soname)
        if package is None:
            problems.append(f"No Debian package is mapped for {soname}")
            continue
        if package not in seen:
            seen.add(package)
            found.append(package)
    if problems:
        raise RuntimeError("\n".join(dict.fromkeys(problems)))
    return found


def _needed_sonames(path: Path) -> list[str]:
    """
    Read the NEEDED entries of one shared library.

    Inputs: path, an ELF file.
    Output: sonames in file order.
    Uses: packages_for_tree.
    Limitations: a missing readelf or a failed read raises RuntimeError.
    """

    result = subprocess.run(
        ["readelf", "-d", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "readelf failed").strip()
        raise RuntimeError(detail[-800:])
    sonames: list[str] = []
    for line in result.stdout.splitlines():
        if "(NEEDED)" in line and "[" in line:
            sonames.append(line.split("[", 1)[1].split("]", 1)[0])
    return sonames


def external_packages(sonames: list[str], bundled: set[str]) -> list[str]:
    """
    Turn sonames that are not inside the package into Debian package names.

    Inputs: sonames, NEEDED entries in first-seen order; bundled, file names shipped under the prefix.
    Output: package names, in first-seen order, without duplicates.
    Uses: packages_for_tree.
    Limitations: a soname in bundled is skipped. An unmapped soname raises RuntimeError. This does not install packages.
    """

    found: list[str] = []
    seen: set[str] = set()
    problems: list[str] = []
    for soname in sonames:
        if soname in bundled:
            continue
        package = package_for_filename(soname)
        if package is None:
            problems.append(f"No Debian package is mapped for {soname}")
            continue
        if package not in seen:
            seen.add(package)
            found.append(package)
    if problems:
        raise RuntimeError("\n".join(dict.fromkeys(problems)))
    return found


def _bundled_names(prefix: Path) -> dict[str, Path]:
    """
    Index shared libraries shipped inside the package by file name.

    Inputs: prefix, the packaged Python directory.
    Output: a map from file name to path. The first file of a given name wins.
    Uses: packages_for_tree when it follows a NEEDED entry into the package.
    Limitations: this indexes file names, which match the sonames Qt and the wheels publish. It does not read ELF headers.
    """

    index: dict[str, Path] = {}
    for path in prefix.rglob("*"):
        if path.is_file() and ".so" in path.name:
            index.setdefault(path.name, path)
    return index


def _entry_files(prefix: Path) -> list[Path]:
    """
    Find the libraries the app and the window plugins load.

    Inputs: prefix, the packaged Python directory.
    Output: existing files, in the order of _APP_LIBRARIES.
    Uses: packages_for_tree.
    Limitations: a missing pattern raises RuntimeError. This does not follow NEEDED entries.
    """

    found: list[Path] = []
    missing: list[str] = []
    for pattern in _APP_LIBRARIES:
        matches = sorted(prefix.glob(pattern))
        if not matches:
            missing.append(pattern)
            continue
        found.extend(matches)
    if missing:
        raise RuntimeError(
            "The package is missing libraries the window needs:\n" + "\n".join(missing)
        )
    return found


def packages_for_tree(prefix: str) -> list[str]:
    """
    Collect Debian dependencies for the libraries the app links.

    Inputs: prefix, the packaged Python directory.
    Output: package names, with the exiftool package last when it was not already present.
    Uses: packaging/linux/build.sh.
    Limitations: readelf supplies the NEEDED entries of the interpreter, Qt widgets, OpenCV, NumPy, rawpy, and the X11 and Wayland plugins. A library inside the prefix is followed instead of mapped. An unmapped system library raises RuntimeError. This does not load the libraries, so the build machine does not need a display stack.
    """

    root = Path(prefix)
    bundled = _bundled_names(root)
    pending = _entry_files(root)
    seen_files: set[str] = set()
    external: list[str] = []
    while pending:
        path = pending.pop(0)
        try:
            key = str(path.resolve())
        except OSError:
            key = str(path)
        if key in seen_files:
            continue
        seen_files.add(key)
        for soname in _needed_sonames(path):
            inside = bundled.get(soname)
            if inside is not None:
                pending.append(inside)
                continue
            external.append(soname)
    found = external_packages(external, set())
    if EXIFTOOL_PACKAGE not in found:
        found.append(EXIFTOOL_PACKAGE)
    return found


def main(argv: list[str] | None = None) -> int:
    """
    Print a Debian Depends list for one packaged Python tree.

    Inputs: argv, a single prefix path.
    Output: 0 after printing the comma-separated packages. 1 when the tree cannot be mapped.
    Uses: packaging/linux/build.sh.
    Limitations: this prints to stdout. Progress and errors go to stderr.
    """

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print("usage: system_depends.py PREFIX", file=sys.stderr)
        return 1
    try:
        print(", ".join(packages_for_tree(args[0])))
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
