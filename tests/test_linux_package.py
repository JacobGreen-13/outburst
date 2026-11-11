"""The Debian package launcher, desktop entry, and library map.

The tests read the files the package installs and check that an ldd transcript becomes Debian package names. They do not build the .deb.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINUX = ROOT / "packaging" / "linux"
_SPEC = importlib.util.spec_from_file_location("outburst_system_depends", LINUX / "system_depends.py")
assert _SPEC is not None and _SPEC.loader is not None
_DEPENDS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DEPENDS)
packages_from_ldd = _DEPENDS.packages_from_ldd


def test_launcher_uses_the_packaged_python() -> None:
    text = (LINUX / "outburst").read_text(encoding="utf-8")
    assert "OUTBURST_PACKAGED=1" in text
    assert text.strip().endswith('exec /opt/outburst/python/bin/python3 -m outburst.main "$@"')


def test_desktop_entry_opens_outburst() -> None:
    text = (LINUX / "outburst.desktop").read_text(encoding="utf-8")
    assert "Name=Outburst" in text
    assert "Exec=outburst %f" in text
    assert "Icon=outburst" in text
    assert "Terminal=false" in text


def test_ldd_keeps_libraries_shipped_in_the_package() -> None:
    text = "\n".join(
        [
            "\tlibQt6Core.so.6 => /opt/outburst/python/lib/libQt6Core.so.6 (0x1)",
            "\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x2)",
            "\tlinux-vdso.so.1 (0x3)",
        ]
    )
    assert packages_from_ldd(text, "/opt/outburst/python") == ["libc6"]


def test_loaded_paths_skip_files_inside_the_package() -> None:
    paths = [
        "/opt/outburst/python/lib/python3.12/site-packages/PySide6/libQt6Core.so.6",
        "/lib/x86_64-linux-gnu/libc.so.6",
        "/lib64/ld-linux-x86-64.so.2",
    ]
    assert _DEPENDS.packages_from_paths(paths, "/opt/outburst/python") == ["libc6"]


def test_versioned_library_names_map_to_packages() -> None:
    paths = [
        "/lib/x86_64-linux-gnu/libz.so.1.3",
        "/usr/lib/locale/locale-archive",
        "/lib/x86_64-linux-gnu/libglib-2.0.so.0.8000.0",
    ]
    assert _DEPENDS.packages_from_paths(paths, "/opt/outburst/python") == [
        "zlib1g",
        "libglib2.0-0t64 | libglib2.0-0",
    ]


def test_xcb_plugin_sonames_become_packages() -> None:
    sonames = [
        "libQt6XcbQpa.so.6",
        "libQt6Gui.so.6",
        "libxcb-cursor.so.0",
        "libxkbcommon-x11.so.0",
        "libwayland-client.so.0",
    ]
    assert _DEPENDS.packages_from_sonames(sonames) == [
        "libxcb-cursor0",
        "libxkbcommon-x11-0",
        "libwayland-client0",
    ]


def test_an_unmapped_system_library_is_reported() -> None:
    text = "\tlibmissing.so.1 => /usr/lib/libmissing.so.1 (0x1)\n"
    with pytest.raises(RuntimeError, match="libmissing.so.1"):
        packages_from_ldd(text, "/opt/outburst/python")
