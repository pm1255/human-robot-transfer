#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode="${1:-cpu}"
case "$mode" in cpu|render) ;; *) echo 'Usage: bash scripts/setup.sh [cpu|render]' >&2; exit 2;; esac
command -v ffmpeg >/dev/null || { echo 'Install ffmpeg first: sudo apt-get install ffmpeg (Ubuntu), or brew install ffmpeg (macOS).' >&2; exit 2; }
command -v ffprobe >/dev/null
"${PYTHON_BIN:-python3}" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e './human2robot[test,vision]'
if [ "$mode" = render ]; then
  .venv/bin/python -m pip install 'mujoco==3.3.7' 'trimesh==5.1.0' 'pycollada==0.9.3' 'fast-simplification==0.2.0' 'numba==0.67.0'
fi
.venv/bin/python tools/run.py models
printf '%s\n' 'Ready. Run: source .venv/bin/activate' 'Help: python tools/run.py --help'
