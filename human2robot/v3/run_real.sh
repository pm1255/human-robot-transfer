#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/workspace/human2robot-v3-20260923
cd "$ROOT/code"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/user/panmiao/workspace/envs/windtunnel/bin/python
"$PY" -c 'import torch,sys;print(sys.executable,torch.__version__,torch.cuda.is_available(),torch.cuda.device_count());assert torch.cuda.device_count()==8'
"$PY" -m unittest v3.test_geometry -v
"$PY" -c 'import json;from pathlib import Path;a=json.loads(Path("../DATA_READY_REAL.json").read_text());assert a["bridge"]["train"]["windows"]==10000'
mkdir -p "$ROOT/training"
mkdir -p "$ROOT/training/code_snapshot_real"
cp -r v3 "$ROOT/training/code_snapshot_real/"
sha256sum v3/*.py > "$ROOT/training/code_sha256_real.txt"
pids=(); i=0
for seed in 17 29 43 71; do
 for condition in bridge robotwin_only; do
  CUDA_VISIBLE_DEVICES="$i" timeout 21600 "$PY" -u -m v3.train --condition "$condition" --seed "$seed" --out "$ROOT/training/seed$seed/$condition" --pre-steps 20000 --fine-steps 10000 > "$ROOT/training/seed${seed}_${condition}.log" 2>&1 &
  pids+=("$!"); i=$((i+1))
 done
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit "$fail"
