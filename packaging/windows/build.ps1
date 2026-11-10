# Build the Windows 11 setup program for Outburst.
# Requires Windows, Python 3.11 or newer, and Inno Setup 6.
# Usage, from anywhere:
#   powershell -ExecutionPolicy Bypass -File packaging\windows\build.ps1

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$Venv = Join-Path $Root "build\windows-venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Dist = Join-Path $Root "dist\Outburst"
$Icon = Join-Path $Root "build\windows\outburst.ico"
$ExifVersion = "13.62"
$ExifUrl = "https://downloads.sourceforge.net/project/exiftool/exiftool-$ExifVersion`_64.zip"

function Invoke-Checked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE"
    }
}

if (-not (Test-Path $Python)) {
    Write-Host "Creating a Windows build environment..."
    $VenvParent = Split-Path $Venv -Parent
    New-Item -ItemType Directory -Force -Path $VenvParent | Out-Null
    if ($env:OUTBURST_PYTHON) {
        Invoke-Checked { & $env:OUTBURST_PYTHON -m venv $Venv }
    } elseif (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv $Venv
        if (-not (Test-Path $Python)) {
            & py -3 -m venv $Venv
        }
    } else {
        Invoke-Checked { & python -m venv $Venv }
    }
    if (-not (Test-Path $Python)) {
        throw "Could not create the build environment. Install Python 3.11 or newer."
    }
}

Write-Host "Installing Outburst and PyInstaller..."
Invoke-Checked { & $Python -m pip install --upgrade pip }
Invoke-Checked { & $Python -m pip install $Root }
Invoke-Checked { & $Python -m pip install "pyinstaller>=6.11" }

Write-Host "Drawing the application icon..."
New-Item -ItemType Directory -Force -Path (Split-Path $Icon -Parent) | Out-Null
Invoke-Checked { & $Python (Join-Path $Root "packaging\windows\make_icon.py") $Icon }
if (-not (Test-Path $Icon)) {
    throw "The icon file was not written: $Icon"
}

Write-Host "Freezing the review window..."
Invoke-Checked { & $Python -m PyInstaller --noconfirm --clean (Join-Path $Root "packaging\windows\outburst.spec") }
if (-not (Test-Path (Join-Path $Dist "Outburst.exe"))) {
    throw "PyInstaller did not write $Dist\Outburst.exe"
}
foreach ($required in @("Qt6Core.dll", "Qt6Gui.dll", "Qt6Widgets.dll", "qwindows.dll")) {
    $found = Get-ChildItem -Path $Dist -Recurse -Filter $required -File | Select-Object -First 1
    if (-not $found) {
        throw "The frozen app is missing $required"
    }
}

Write-Host "Adding ExifTool $ExifVersion..."
$ExifZip = Join-Path $Root "build\exiftool-$ExifVersion.zip"
$ExifExtract = Join-Path $Root "build\exiftool-$ExifVersion"
Invoke-WebRequest -Uri $ExifUrl -OutFile $ExifZip -UserAgent "Outburst-installer"
$ExifSize = (Get-Item $ExifZip).Length
if ($ExifSize -lt 1000000) {
    throw "The ExifTool download was $ExifSize bytes. The address may have returned a web page instead of the zip."
}
if (Test-Path $ExifExtract) {
    Remove-Item -Recurse -Force $ExifExtract
}
Expand-Archive -Path $ExifZip -DestinationPath $ExifExtract -Force
$Tool = Get-ChildItem -Path $ExifExtract -Recurse -File -Filter "exiftool*.exe" | Select-Object -First 1
if (-not $Tool) {
    throw "The ExifTool archive did not contain an executable."
}
Copy-Item $Tool.FullName (Join-Path $Dist "exiftool.exe") -Force
$FilesDir = Get-ChildItem -Path $ExifExtract -Recurse -Directory | Where-Object { $_.Name -eq "exiftool_files" } | Select-Object -First 1
if (-not $FilesDir) {
    throw "The ExifTool archive did not contain exiftool_files."
}
$FilesDest = Join-Path $Dist "exiftool_files"
if (Test-Path $FilesDest) {
    Remove-Item -Recurse -Force $FilesDest
}
Copy-Item $FilesDir.FullName $FilesDest -Recurse -Force

$LicenseDir = Join-Path $Dist "Licenses"
New-Item -ItemType Directory -Force -Path $LicenseDir | Out-Null
Copy-Item (Join-Path $Root "LICENSE") (Join-Path $LicenseDir "Outburst-MIT.txt") -Force
Copy-Item (Join-Path $Root "packaging\windows\THIRD-PARTY.txt") (Join-Path $LicenseDir "THIRD-PARTY.txt") -Force

$Version = & $Python -c "from outburst import __version__; print(__version__)"
if ($LASTEXITCODE -ne 0 -or -not $Version) {
    throw "Could not read the Outburst version."
}
$Version = $Version.Trim()

$IsccCandidates = @(
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $Iscc) {
    $Command = Get-Command ISCC.exe -ErrorAction SilentlyContinue
    if ($Command) {
        $Iscc = $Command.Source
    }
}
if (-not $Iscc) {
    throw "Inno Setup 6 is required. Install it with: choco install innosetup"
}

Write-Host "Compiling the installer..."
Invoke-Checked { & $Iscc "/DMyAppVersion=$Version" (Join-Path $Root "packaging\windows\Outburst.iss") }
$Setup = Join-Path $Root "dist\installer\Outburst-Setup-$Version.exe"
if (-not (Test-Path $Setup)) {
    throw "Inno Setup did not write $Setup"
}
$Hash = Get-FileHash $Setup -Algorithm SHA256
$HashLine = "$($Hash.Hash)  $(Split-Path $Setup -Leaf)"
Set-Content -Path (Join-Path $Root "dist\installer\SHA256.txt") -Value $HashLine -Encoding ascii
Write-Host "Wrote $Setup"
Write-Host $HashLine
