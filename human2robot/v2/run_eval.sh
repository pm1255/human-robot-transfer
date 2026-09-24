#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/workspace/human2robot-v2-20260923
cd "$ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/user/chenzilong/envs/robotwin/bin/python
# Eight independent models; two simulator processes per GPU.
pids=()
worker=0
for seed in 17 29; do
 for condition in robot_only explicit implicit_joint shuffled; do
  gpu=$((worker % 4))
  CUDA_VISIBLE_DEVICES="$gpu" timeout 5400 "$PY" -u -m v2.run_eval --root "$ROOT" --training-seed "$seed" --conditions "$condition" > "$ROOT/eval_absolute_${seed}_${condition}.log" 2>&1 &
  pids+=("$!")
  worker=$((worker+1))
 done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
