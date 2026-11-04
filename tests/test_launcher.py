"""The one-click launcher finds Python and does not install during a dry run."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        args,
        cwd=ROOT,
        env=merged,
        capture_output=True,
        text=True,
        check=False,
    )


def test_launcher_syntax() -> None:
    script = _run(["bash", "-n", "outburst.sh"])
    assert script.returncode == 0, script.stderr
    command = _run(["bash", "-n", "Outburst.command"])
    assert command.returncode == 0, command.stderr


def test_launcher_help() -> None:
    result = _run(["bash", "outburst.sh", "--help"])
    assert result.returncode == 0
    assert "subject" in result.stdout
    assert ".venv" in result.stdout


def test_launcher_dry_run_selects_python() -> None:
    result = _run(["bash", "outburst.sh"], {"OUTBURST_DRY_RUN": "1"})
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert values["extras"] == ""
    assert values["folder"] == ""
    version = subprocess.run(
        [values["python"], "-c", "import sys; print(sys.version_info >= (3, 11))"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert version.stdout.strip() == "True"


def test_launcher_dry_run_accepts_subject_and_folder() -> None:
    result = _run(
        ["bash", "outburst.sh", "subject", "examples/sample_burst"],
        {"OUTBURST_DRY_RUN": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "extras=[subject]" in result.stdout
    assert "folder=examples/sample_burst" in result.stdout


def test_launcher_rejects_a_python_older_than_3_11(tmp_path: Path) -> None:
    stub = tmp_path / "python"
    stub.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    stub.chmod(0o755)
    result = _run(["bash", "outburst.sh"], {"OUTBURST_DRY_RUN": "1", "OUTBURST_PYTHON": str(stub)})
    assert result.returncode == 1
    assert "OUTBURST_PYTHON" in result.stderr
