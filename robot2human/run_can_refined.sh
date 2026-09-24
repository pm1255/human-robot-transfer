#!/usr/bin/env bash
set -euo pipefail
R2H_ROOT=/user/panmiao/workspace/robot2human-vace-20260924
cd "$R2H_ROOT"
export PYTHONPATH="$R2H_ROOT/deps:$R2H_ROOT/Wan2.1-main:$R2H_ROOT/sam2-code:/user/panmiao/workspace/human2robot-wan-v4-20260924/deps"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=/user/panmiao/workspace/envs/windtunnel/bin/python
"$PY" -u refine_can_mask.py > refine_can_mask.log 2>&1
if test -d results/1.3B/bridge_bridge_045160/seed2026; then mv results/1.3B/bridge_bridge_045160/seed2026 results/1.3B/bridge_bridge_045160/seed2026_initial_mask; fi
"$PY" -u generate.py --model 1.3B --steps 40 --size 512 --only bridge_bridge_045160 > generation_can_refined.log 2>&1
