#!/usr/bin/env bash
set -euo pipefail
R2H_ROOT=/user/panmiao/workspace/robot2human-vace-20260924
cd "$R2H_ROOT"
export PYTHONPATH="$R2H_ROOT/deps:$R2H_ROOT/Wan2.1-main:$R2H_ROOT/sam2-code:/user/panmiao/workspace/human2robot-wan-v4-20260924/deps"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PY=/user/panmiao/workspace/envs/windtunnel/bin/python
for i in $(seq 1 60); do
  if test -f "$R2H_ROOT/ASSETS_READY.json" && test -f "$R2H_ROOT/Wan2.1-VACE-1.3B_READY.json"; then break; fi
  sleep 20
done
test -f "$R2H_ROOT/ASSETS_READY.json"
test -f "$R2H_ROOT/Wan2.1-VACE-1.3B_READY.json"
command -v ffmpeg > ffmpeg_path.txt || true
"$PY" -u segment.py > masks.log 2>&1
"$PY" -u generate.py --model 1.3B --steps 40 --size 512 > generation_1.3B.log 2>&1
