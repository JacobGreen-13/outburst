@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

if /I "%~1"=="-h" goto :help
if /I "%~1"=="--help" goto :help
if /I "%~1"=="/?" goto :help

rem "subject" remains accepted so an older shortcut still opens a folder.
if /I "%~1"=="subject" (
  shift
)
set "FOLDER="
if not "%~1"=="" set "FOLDER=%~1"

set "PYTHON="
for %%V in (3.13 3.12 3.11) do (
  py -%%V -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
  if not errorlevel 1 (
    set "PYTHON=py -%%V"
    goto :have_python
  )
)
python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PYTHON=python"
  goto :have_python
)
echo Outburst needs Python 3.11 or newer.
echo Install it from https://www.python.org/downloads/ and check "Add python.exe to PATH".
pause
exit /b 1

:have_python
if "%OUTBURST_DRY_RUN%"=="1" (
  echo python=!PYTHON!
  echo extras=
  echo folder=!FOLDER!
  exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating a Python environment...
  %PYTHON% -m venv .venv
  if errorlevel 1 goto :failed
)

".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)" >nul 2>&1
if errorlevel 1 (
  echo The environment in .venv is older than Python 3.11.
  echo Delete the .venv folder and run this script again.
  goto :failed
)

set "NEED_INSTALL=0"
if not exist ".venv\Scripts\outburst.exe" set "NEED_INSTALL=1"
if not exist ".venv\.outburst-stamp" set "NEED_INSTALL=1"
if exist ".venv\.outburst-stamp" (
  ".venv\Scripts\python.exe" -c "import sys; from pathlib import Path; stamp=Path(r'.venv/.outburst-stamp'); src=Path('pyproject.toml'); sys.exit(0 if stamp.stat().st_mtime >= src.stat().st_mtime else 1)"
  if errorlevel 1 set "NEED_INSTALL=1"
)

if "!NEED_INSTALL!"=="1" (
  echo Installing Outburst.
  echo Subject and bird eye detection stay off until you install them in the window.
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  if errorlevel 1 goto :failed
  ".venv\Scripts\python.exe" -m pip install -e .
  if errorlevel 1 goto :failed
  echo installed> ".venv\.outburst-stamp"
)

".venv\Scripts\python.exe" -c "import PySide6" >nul 2>&1
if errorlevel 1 (
  echo Outburst could not load its window toolkit.
  ".venv\Scripts\python.exe" -c "import PySide6"
  goto :failed
)

where exiftool >nul 2>&1
if errorlevel 1 echo Note: exiftool is not installed. Burst times still work. It is more reliable for CR3.

echo Opening Outburst...
if exist ".venv\Scripts\outburst-gui.exe" (
  set "APP=.venv\Scripts\outburst-gui.exe"
) else (
  set "APP=.venv\Scripts\outburst.exe"
)
if not "!FOLDER!"=="" (
  start "" "!APP!" "!FOLDER!"
) else (
  start "" "!APP!"
)
exit /b 0

:help
echo Usage: Outburst.bat [folder]
echo        Outburst.bat subject [folder]
echo.
echo The first run creates .venv, installs Outburst, and opens the window.
echo Later runs open the window. Subject detection and bird eye detection
echo stay off until you install them in the window.
echo The subject argument is still accepted.
exit /b 0

:failed
echo.
echo Outburst did not start.
pause
exit /b 1
