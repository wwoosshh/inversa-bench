#!/usr/bin/env python3
"""Inversa one-command launcher — run me with ANY Python: `python run.py` (or double-click start.bat).

A fresh clone needs no manual setup: this finds a Python 3.11+ interpreter, builds a local virtual
environment (.venv), installs Inversa + its dependencies into it, and opens the GUI in your browser.
Re-runs are instant (setup is skipped once done). Stdlib-only and does NOT import inversa, so it works
before anything is installed.

Env overrides (mainly for testing): INVERSA_VENV_DIR (venv location), INVERSA_SETUP_ONLY=1 (skip the
GUI launch — set up only).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = Path(os.environ.get("INVERSA_VENV_DIR") or (ROOT / ".venv"))
MIN_PY = (3, 11)


def venv_python(venv_dir: Path) -> Path:
    """Path to the python executable inside a venv, per platform."""
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _probe_version(cmd) -> tuple | None:
    """Run `cmd` (a list) as a Python and return its (major, minor), or None if it isn't usable."""
    try:
        out = subprocess.run(cmd + ["-c", "import sys;print('%d %d'%sys.version_info[:2])"],
                             capture_output=True, text=True, timeout=25)
        if out.returncode != 0:
            return None
        a, b = (int(x) for x in out.stdout.split())
        return (a, b)
    except Exception:
        return None


def find_base_python():
    """A command (list) for a Python >= MIN_PY: prefer the Windows `py` launcher, then versioned
    names, then python3/python. Returns None if none qualifies."""
    candidates = []
    if os.name == "nt":
        candidates += [["py", f"-{v}"] for v in ("3.13", "3.12", "3.11")]
    candidates += [[n] for n in ("python3.13", "python3.12", "python3.11", "python3", "python")]
    for cmd in candidates:
        v = _probe_version(cmd)
        if v and v >= MIN_PY:
            return cmd
    return None


def ensure_venv() -> Path:
    """Return the venv's python, (re)creating the venv if missing or broken."""
    vpy = venv_python(VENV)
    if VENV.exists() and (_probe_version([str(vpy)]) or (0, 0)) >= MIN_PY:
        return vpy
    if VENV.exists():
        print(f"[setup] {VENV.name} is unusable on this machine — recreating it…", flush=True)
        shutil.rmtree(VENV, ignore_errors=True)
    base = find_base_python()
    if base is None:
        sys.exit("\nInversa needs Python 3.11+, but none was found on this system.\n"
                 "Install it from https://www.python.org/downloads/ (tick 'Add to PATH'), then "
                 "re-run.\n")
    print(f"[setup] creating virtual environment with: {' '.join(base)}", flush=True)
    subprocess.check_call(base + ["-m", "venv", str(VENV)])
    return venv_python(VENV)


def ensure_installed(vpy: Path) -> None:
    """Install Inversa into the venv once (skip if already importable there)."""
    if subprocess.run([str(vpy), "-c", "import inversa"], capture_output=True).returncode == 0:
        return
    print("[setup] installing Inversa + dependencies (one-time, ~1 min)…", flush=True)
    subprocess.check_call([str(vpy), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    subprocess.check_call([str(vpy), "-m", "pip", "install", "-e", "."], cwd=str(ROOT))


def main() -> None:
    vpy = ensure_venv()
    ensure_installed(vpy)
    if os.environ.get("INVERSA_SETUP_ONLY") == "1":
        print("[setup] done (INVERSA_SETUP_ONLY).", flush=True)
        return
    print("[launch] opening the Inversa GUI in your browser…", flush=True)
    os.chdir(ROOT)  # so data/banks + data/results resolve
    raise SystemExit(subprocess.call([str(vpy), "-m", "inversa.gui"]))


if __name__ == "__main__":
    main()
