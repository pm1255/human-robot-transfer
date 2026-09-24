#!/usr/bin/env bash
set -euo pipefail
R2H_ROOT=/user/panmiao/workspace/robot2human-vace-20260924
cd "$R2H_ROOT"
export PYTHONPATH="$R2H_ROOT/deps:$R2H_ROOT/Wan2.1-main:$R2H_ROOT/sam2-code:/user/panmiao/workspace/human2robot-wan-v4-20260924/deps"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=/user/panmiao/workspace/envs/windtunnel/bin/python
"$PY" -u segment_expanded.py > masks_expanded.log 2>&1
"$PY" -u generate_expanded.py --variant handroom --seeds 2026 2027 > generation_handroom.log 2>&1
