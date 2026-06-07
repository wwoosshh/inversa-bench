#!/usr/bin/env sh
# Inversa — run to set up (first run) and launch the GUI. No prior setup needed.
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then exec python3 run.py "$@"; else exec python run.py "$@"; fi
