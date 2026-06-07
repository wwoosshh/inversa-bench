"""The one-command launcher (run.py) bootstraps a fresh clone: find Python 3.11+, build a venv,
install, open the GUI. It must be runnable by ANY Python (stdlib only, no inversa import), so its
pure helpers are loaded from the repo-root file and tested directly.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

_RUN = Path(__file__).resolve().parents[1] / "run.py"


def _load():
    spec = importlib.util.spec_from_file_location("inversa_run", _RUN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_py_exists():
    assert _RUN.exists()


def test_venv_python_path_is_platform_correct():
    run = _load()
    p = run.venv_python(Path("/proj/.venv"))
    s = str(p).replace("\\", "/")
    assert s.endswith("Scripts/python.exe" if os.name == "nt" else "bin/python")


def test_min_python_is_311():
    run = _load()
    assert run.MIN_PY == (3, 11)


def test_no_toplevel_inversa_import():
    # run.py must not IMPORT the package it installs (chicken-and-egg). A "import inversa" string
    # handed to a subprocess is fine; a real top-level import statement is not.
    import re
    src = _RUN.read_text(encoding="utf-8")
    assert not re.search(r"(?m)^\s*(import inversa|from inversa\b)", src)
