"""FastSim HL-LHC Launcher -- double-click to run.

This script is compiled into FastSim.exe via PyInstaller.  It ensures
the virtual environment exists, installs the package and opens the
Streamlit dashboard in the default browser.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
SCRIPTS = VENV / "Scripts"  # Windows
BIN = VENV / "bin"          # Linux/Mac


def find_python() -> Path:
    """Return the venv Python path, creating the venv if needed."""
    if SCRIPTS.exists():
        return SCRIPTS / "python.exe"
    if BIN.exists():
        return BIN / "python"

    # Create venv.
    subprocess.check_call(
        [sys.executable, "-m", "venv", str(VENV)],
        cwd=str(ROOT),
    )
    if SCRIPTS.exists():
        return SCRIPTS / "python.exe"
    return BIN / "python"


def ensure_installed(python: Path) -> None:
    """Install the package if not already present."""
    result = subprocess.run(
        [str(python), "-c", "import fastsim_tt4a"],
        capture_output=True,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "-q", "-e", ".[dev,ui]"],
            cwd=str(ROOT),
        )


def launch_dashboard(python: Path) -> None:
    """Start the Streamlit dashboard."""
    streamlit_script = ROOT / "src" / "fastsim_tt4a" / "dashboard.py"
    subprocess.Popen(
        [str(python), "-m", "streamlit", "run", str(streamlit_script)],
        cwd=str(ROOT),
    )


def main() -> None:
    python = find_python()
    ensure_installed(python)
    launch_dashboard(python)


if __name__ == "__main__":
    main()
