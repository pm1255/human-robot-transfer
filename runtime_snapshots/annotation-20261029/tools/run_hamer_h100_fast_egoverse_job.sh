#!/usr/bin/env bash
# Backward-compatible job entry. "fast" no longer means point-cloud rendering.
set -euo pipefail
ROOT=${HAMER_SHARED_ROOT:-/user/panmiao/human-robot-transfer}
exec bash "$ROOT/tools/run_hamer_surface_job.sh"
