#!/usr/bin/env bash
# Current EgoVerse entry point; supersedes the historical scatter/hull renderer.
set -euo pipefail
ROOT=${HAMER_SHARED_ROOT:-/user/panmiao/human-robot-transfer}
exec bash "$ROOT/tools/run_hamer_surface_job.sh"
