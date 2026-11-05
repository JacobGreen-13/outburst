#!/usr/bin/env bash
# Install Outburst into .venv on the first run, then open the window.
# macOS: double-click Outburst.command. Linux: ./outburst.sh or Outburst.desktop.
set -euo pipefail

SOURCE="${BASH_SOURCE[0]}"
ROOT="$(cd "$(dirname "$SOURCE")" && pwd)"
cd "$ROOT"

# A double-clicked Terminal starts with a very short PATH.
export PATH="/opt/homebrew/bin:/usr/local/bin:${HOME}/.local/bin:${PATH:-/usr/bin:/bin}"

usage() {
  cat <<'EOF'
Usage: ./outburst.sh [folder]
       ./outburst.sh subject [folder]

The first run creates .venv, installs Outburst, and opens the window.
Later runs open the window. Subject detection and bird eye detection
are included. The first run downloads PyTorch and can take several
minutes. The subject argument is still accepted.
EOF
}

log() {
  printf '%s\n' "$*" >&2
}

pause_on_failure() {
  if [[ -t 0 ]]; then
    read -r -p "Press Return to close..." _
  fi
}

die() {
  log "$*"
  pause_on_failure
  exit 1
}

python_ok() {
  local bin="$1"
  [[ -n "$bin" && -x "$bin" ]] || return 1
  "$bin" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

find_python() {
  local name prefix version candidate
  for version in 3.13 3.12 3.11; do
    for prefix in /opt/homebrew /usr/local "${HOME}/.local"; do
      candidate="${prefix}/bin/python${version}"
      if python_ok "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
      fi
    done
    candidate="/Library/Frameworks/Python.framework/Versions/${version}/bin/python3"
    if python_ok "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  for name in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$name" >/dev/null 2>&1; then
      candidate="$(command -v "$name")"
      if python_ok "$candidate"; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  for candidate in \
    "${HOME}/miniforge3/bin/python3" \
    "${HOME}/miniconda3/bin/python3" \
    "${HOME}/anaconda3/bin/python3"
  do
    if python_ok "$candidate"; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

# "subject" remains accepted so an older shortcut still opens a folder.
if [[ "${1:-}" == "subject" ]]; then
  shift
fi

if [[ -n "${OUTBURST_PYTHON:-}" ]]; then
  python_ok "$OUTBURST_PYTHON" || die "OUTBURST_PYTHON is not Python 3.11 or newer: $OUTBURST_PYTHON"
  PYTHON="$OUTBURST_PYTHON"
else
  PYTHON="$(find_python)" || die "Outburst needs Python 3.11 or newer. Install it from https://www.python.org/downloads/ or with: brew install python@3.12"
fi

if [[ "${OUTBURST_DRY_RUN:-}" == "1" ]]; then
  printf 'python=%s\n' "$PYTHON"
  printf 'extras=[subject]\n'
  if [[ $# -gt 0 ]]; then
    printf 'folder=%s\n' "$1"
  else
    printf 'folder=\n'
  fi
  exit 0
fi

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
STAMP="$VENV/.outburst-stamp"
SUBJECT_STAMP="$VENV/.outburst-subject"

if [[ ! -x "$PY" ]]; then
  log "Creating a Python environment..."
  "$PYTHON" -m venv "$VENV" || die "Could not create .venv. Install Python 3.11 or newer and run this script again."
fi

"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
  || die "The environment in .venv is older than Python 3.11. Delete .venv and run this script again."

EXTRAS="[subject]"

NEED_INSTALL=0
if [[ ! -x "$VENV/bin/outburst" || ! -f "$STAMP" ]]; then
  NEED_INSTALL=1
elif [[ "$ROOT/pyproject.toml" -nt "$STAMP" ]]; then
  NEED_INSTALL=1
elif [[ ! -f "$SUBJECT_STAMP" ]]; then
  NEED_INSTALL=1
elif ! "$PY" -c "import ultralytics" >/dev/null 2>&1; then
  NEED_INSTALL=1
fi

if [[ "$NEED_INSTALL" == "1" ]]; then
  version="$("$PYTHON" -c 'import platform, sys; print(sys.version.split()[0], platform.machine())')"
  log "Installing Outburst with Python ${version}."
  log "Subject and bird eye detection are included. This downloads PyTorch and can take several minutes."
  "$PY" -m pip install --upgrade pip || log "Continuing with the current pip."
  "$PY" -m pip install -e ".${EXTRAS}" || die "Installation failed. Check the messages above and run this script again."
  touch "$STAMP"
  if [[ -n "$EXTRAS" ]]; then
    touch "$SUBJECT_STAMP"
  fi
fi

if ! "$PY" -c "import PySide6" >/dev/null 2>&1; then
  log "Outburst could not load its window toolkit."
  if [[ "$(uname -s)" == "Linux" ]]; then
    log "On Debian and Ubuntu, install libegl1 and run this script again."
  fi
  "$PY" -c "import PySide6" || true
  die "The window toolkit is not usable yet."
fi

if ! command -v exiftool >/dev/null 2>&1; then
  log "Note: exiftool is not installed. Burst times still work. It is more reliable for CR3."
fi

LOG_FILE="${TMPDIR:-/tmp}/outburst-launch.log"
: >"$LOG_FILE"
log "Opening Outburst..."
if [[ $# -gt 0 ]]; then
  nohup "$VENV/bin/outburst" "$@" >>"$LOG_FILE" 2>&1 &
else
  nohup "$VENV/bin/outburst" >>"$LOG_FILE" 2>&1 &
fi
pid=$!
still_running=0
for _ in 1 2 3 4 5 6 7 8; do
  if ! kill -0 "$pid" 2>/dev/null; then
    still_running=0
    break
  fi
  still_running=1
  sleep 0.5
done
if [[ "$still_running" != "1" ]]; then
  wait "$pid" || true
  log "Outburst did not stay open. The last lines of the log:"
  tail -n 30 "$LOG_FILE" >&2 || true
  die "See ${LOG_FILE}"
fi
disown || true
log "Outburst is open. This window can be closed."
