#!/usr/bin/env bash
set -euo pipefail
R2H_ROOT=/user/panmiao/workspace/robot2human-vace-20260924
cd "$R2H_ROOT"
export PYTHONPATH="$R2H_ROOT/deps:$R2H_ROOT/Wan2.1-main:$R2H_ROOT/sam2-code:/user/panmiao/workspace/human2robot-wan-v4-20260924/deps"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=/user/panmiao/workspace/envs/windtunnel/bin/python
test -f "$R2H_ROOT/Wan2.1-VACE-14B_READY.json"
"$PY" -u generate.py --model 14B --steps 40 --size 512 > generation_14B.log 2>&1
