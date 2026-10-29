#!/usr/bin/env bash
# Backward-compatible job entry. "fast" no longer means point-cloud rendering.
set -euo pipefail
ROOT=${HAMER_SHARED_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
exec bash "$ROOT/tools/run_hamer_surface_job.sh"
