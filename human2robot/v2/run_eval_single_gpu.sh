#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/workspace/human2robot-v2-20260923
cd "$ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
PY=/user/chenzilong/envs/robotwin/bin/python
pids=()
for seed in 17 29; do
 for group in robot_only,explicit implicit_joint,shuffled; do
  timeout 9000 "$PY" -u -m v2.run_eval --root "$ROOT" --training-seed "$seed" --conditions "$group" > "$ROOT/eval_single_${seed}_${group}.log" 2>&1 &
  pids+=("$!")
 done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
