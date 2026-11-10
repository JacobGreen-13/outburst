# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the Windows folder build.

The frozen app contains the core review window. Ultralytics, PyTorch, and the YOLO weights are excluded. The install button downloads a private Python and installs them there. The Outburst sources are shipped so that Python can run outburst.detect_worker. ExifTool is copied next to Outburst.exe by packaging/windows/build.ps1 after this spec runs.
"""

import re
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

root = Path(SPECPATH).resolve().parents[1]
version_match = re.search(
    r'__version__ = "([^"]+)"',
    (root / "src" / "outburst" / "__init__.py").read_text(encoding="utf-8"),
)
version = version_match.group(1) if version_match else "0.1.0"
version_parts = [int(piece) if piece.isdigit() else 0 for piece in version.split(".")]
while len(version_parts) < 4:
    version_parts.append(0)
version_tuple = tuple(version_parts[:4])

icon_path = root / "build" / "windows" / "outburst.ico"
version_file = root / "build" / "windows" / "file_version_info.txt"
version_file.parent.mkdir(parents=True, exist_ok=True)
version_file.write_text(
    "\n".join(
        [
            "VSVersionInfo(",
            "  ffi=FixedFileInfo(",
            f"    filevers={version_tuple},",
            f"    prodvers={version_tuple},",
            "    mask=0x3f,",
            "    flags=0x0,",
            "    OS=0x40004,",
            "    fileType=0x1,",
            "    subtype=0x0,",
            "    date=(0, 0)",
            "  ),",
            "  kids=[",
            "    StringFileInfo([StringTable('040904B0', [",
            "      StringStruct('CompanyName', 'Outburst contributors'),",
            "      StringStruct('FileDescription', 'Local burst culling for wildlife and nature photographers'),",
            f"      StringStruct('FileVersion', '{version}'),",
            "      StringStruct('InternalName', 'Outburst'),",
            "      StringStruct('OriginalFilename', 'Outburst.exe'),",
            "      StringStruct('ProductName', 'Outburst'),",
            f"      StringStruct('ProductVersion', '{version}'),",
            "      StringStruct('LegalCopyright', 'Copyright (c) 2026 Outburst contributors')",
            "    ])]),",
            "    VarFileInfo([VarStruct('Translation', [1033, 1200])])",
            "  ]",
            ")",
            "",
        ]
    ),
    encoding="utf-8",
)

datas: list = [(str(root / "src" / "outburst"), "detection_src/outburst")]
binaries: list = []
hiddenimports = collect_submodules("outburst")
for package in ("rawpy", "cv2"):
    package_datas, package_binaries, package_hidden = collect_all(package)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hidden)

a = Analysis(
    [str(root / "src" / "outburst" / "main.py")],
    pathex=[str(root / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["ultralytics", "torch", "torchvision", "tensorflow", "matplotlib", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Outburst",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(icon_path),
    version=str(version_file),
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Outburst",
)
