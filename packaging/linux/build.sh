#!/usr/bin/env bash
# Build a .deb for Debian and Ubuntu.
# Usage, from anywhere:
#   packaging/linux/build.sh
# The package is written to dist/installer/outburst_<version>_<arch>.deb.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_VERSION="3.12.16"
PYTHON_STAMP="20261110"
case "$(uname -m)" in
  x86_64) DEFAULT_TARGET="x86_64-unknown-linux-gnu" ;;
  aarch64|arm64) DEFAULT_TARGET="aarch64-unknown-linux-gnu" ;;
  *)
    echo "This build supports x86_64 and aarch64. This machine is $(uname -m)." >&2
    exit 1
    ;;
esac
TARGET="${OUTBURST_PYTHON_TARGET:-$DEFAULT_TARGET}"
case "$TARGET" in
  x86_64-unknown-linux-gnu) ARCH="amd64" ;;
  aarch64-unknown-linux-gnu) ARCH="arm64" ;;
  *)
    echo "OUTBURST_PYTHON_TARGET must be x86_64-unknown-linux-gnu or aarch64-unknown-linux-gnu." >&2
    exit 1
    ;;
esac

declare -A PYTHON_SHA256=(
  [x86_64-unknown-linux-gnu]="6a9e62dad033790aef722dbd370e0ce76ce17b73627ba3ae679f537049350b39"
  [aarch64-unknown-linux-gnu]="b6a41defec4975d94b4ff1950236e7383e5c02ba40ea48c9ec33c66087a35aab"
)

ASSET="cpython-${PYTHON_VERSION}+${PYTHON_STAMP}-${TARGET}-install_only_stripped.tar.gz"
URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_STAMP}/${ASSET}"
CACHE="$ROOT/build/linux-cache"
STAGE="$ROOT/build/linux-stage"
PREFIX="$STAGE/opt/outburst/python"

rm -rf "$STAGE"
mkdir -p "$CACHE" "$STAGE/opt/outburst" "$STAGE/DEBIAN" "$ROOT/dist/installer"

ARCHIVE="$CACHE/$ASSET"
if [[ ! -f "$ARCHIVE" ]]; then
  echo "Downloading CPython ${PYTHON_VERSION} for ${TARGET}..."
  curl --fail --location --retry 3 --output "$ARCHIVE.partial" "$URL"
  mv "$ARCHIVE.partial" "$ARCHIVE"
fi
echo "${PYTHON_SHA256[$TARGET]}  $ARCHIVE" | sha256sum --check --status

echo "Unpacking CPython..."
tar -xzf "$ARCHIVE" -C "$STAGE/opt/outburst"
if [[ ! -x "$PREFIX/bin/python3" ]]; then
  echo "The CPython archive did not contain python/bin/python3." >&2
  exit 1
fi

echo "Installing Outburst into the package..."
"$PREFIX/bin/python3" -m ensurepip --upgrade
"$PREFIX/bin/python3" -m pip install --upgrade pip
"$PREFIX/bin/python3" -m pip install "$ROOT"
# The review window uses Qt widgets. The extra PySide6 wheels pull in a browser engine and database plugins.
"$PREFIX/bin/python3" -m pip uninstall -y pyside6 PySide6_WebEngine PySide6_Addons PySide6_Pdf
"$PREFIX/bin/python3" - <<'PY'
import pathlib
import subprocess

prefix = pathlib.Path("build/linux-stage/opt/outburst/python")
present: set[str] = set()
for path in prefix.rglob("*"):
    if not path.is_file() or ".so" not in path.name:
        continue
    present.add(path.name)
    listed = subprocess.run(
        ["readelf", "-d", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        continue
    for line in listed.stdout.splitlines():
        if "(SONAME)" in line and "[" in line:
            present.add(line.split("[", 1)[1].split("]", 1)[0])
removed = 0
for path in list(prefix.rglob("*.so")):
    listed = subprocess.run(
        ["readelf", "-d", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        continue
    needed = [
        line.split("[", 1)[1].split("]", 1)[0]
        for line in listed.stdout.splitlines()
        if "(NEEDED)" in line and "[" in line
    ]
    if any(name.startswith("libQt") and name not in present for name in needed):
        path.unlink()
        removed += 1
print(f"Removed {removed} plugins that need a Qt library this package does not ship.")
PY
find "$PREFIX" -type d -name __pycache__ -exec rm -rf {} + || true
rm -rf "$PREFIX/lib/python3.12/test" "$PREFIX/lib/python3.12/config-"* || true

"$PREFIX/bin/python3" - <<'PY'
import pathlib
import stat
stage = pathlib.Path("build/linux-stage/opt/outburst/python/bin")
target = "#!/opt/outburst/python/bin/python3\n"
for path in stage.iterdir():
    if not path.is_file() or path.is_symlink():
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        continue
    if not text.startswith("#!") or "python" not in text.splitlines()[0]:
        continue
    lines = text.splitlines(keepends=True)
    lines[0] = target
    path.write_text("".join(lines), encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
PY

echo "Checking that the window toolkit imports..."
if ! QT_QPA_PLATFORM=offscreen "$PREFIX/bin/python3" -c "import outburst, PySide6.QtWidgets, cv2, numpy, rawpy, exifread"; then
  echo "The build machine is missing a library the window loads. Install the packages in packaging/linux/build-packages.txt, then run this script again." >&2
  exit 1
fi

QT_QPA_PLATFORM=offscreen "$PREFIX/bin/python3" \
  "$ROOT/packaging/linux/make_icon.py" \
  "$STAGE/usr/share/icons/hicolor/256x256/apps/outburst.png"

install -D -m 0755 "$ROOT/packaging/linux/outburst" "$STAGE/usr/bin/outburst"
install -D -m 0644 "$ROOT/packaging/linux/outburst.desktop" "$STAGE/usr/share/applications/outburst.desktop"
install -D -m 0644 "$ROOT/packaging/linux/copyright" "$STAGE/usr/share/doc/outburst/copyright"
install -D -m 0644 "$ROOT/packaging/linux/THIRD-PARTY.txt" "$STAGE/usr/share/doc/outburst/THIRD-PARTY.txt"
install -D -m 0644 "$ROOT/LICENSE" "$STAGE/usr/share/doc/outburst/LICENSE"
install -m 0755 "$ROOT/packaging/linux/postinst" "$STAGE/DEBIAN/postinst"
install -m 0755 "$ROOT/packaging/linux/postrm" "$STAGE/DEBIAN/postrm"

VERSION="$("$PREFIX/bin/python3" -c "import outburst; print(outburst.__version__)")"
DEPENDS="$("$PREFIX/bin/python3" "$ROOT/packaging/linux/system_depends.py" "$PREFIX")"
DATE_STAMP="$(date -R)"

cat >"$STAGE/usr/share/doc/outburst/changelog.Debian" <<EOF
outburst (${VERSION}) unstable; urgency=medium

  * Debian package for Outburst ${VERSION}.

 -- Outburst contributors <outburst-contributors@users.noreply.github.com>  ${DATE_STAMP}
EOF
gzip -n -9 -f "$STAGE/usr/share/doc/outburst/changelog.Debian"
INSTALLED_SIZE="$(du -sk "$STAGE/opt" "$STAGE/usr" | awk '{sum += $1} END {print sum}')"

cat >"$STAGE/DEBIAN/control" <<EOF
Package: outburst
Version: ${VERSION}
Section: graphics
Priority: optional
Architecture: ${ARCH}
Maintainer: Outburst contributors <outburst-contributors@users.noreply.github.com>
Installed-Size: ${INSTALLED_SIZE}
Depends: ${DEPENDS}
Homepage: https://github.com/JacobGreen-13/outburst
Description: Local burst culling for wildlife and nature photographers
 Outburst groups frames shot a fraction of a second apart, scores each one,
 and opens on the frame it would keep. Original photographs stay untouched.
 Subject and bird eye detection stay off until they are installed from the window.
EOF

(
  cd "$STAGE"
  find opt usr -type f -print0 | sort -z | xargs -0 md5sum
) >"$STAGE/DEBIAN/md5sums"

PACKAGE="$ROOT/dist/installer/outburst_${VERSION}_${ARCH}.deb"
dpkg-deb --root-owner-group --build "$STAGE" "$PACKAGE"
sha256sum "$PACKAGE" | tee "$ROOT/dist/installer/SHA256.txt"
echo "Wrote $PACKAGE"
