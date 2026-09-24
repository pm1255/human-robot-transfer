#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/workspace/human2robot-wan-v4-20260924
PY=/user/panmiao/workspace/envs/windtunnel/bin/python
cd "$ROOT/code"
export PYTHONPATH="$ROOT/deps:$ROOT/code" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
"$PY" -c 'import sys,torch;print(sys.executable,torch.__version__,torch.cuda.device_count());assert torch.cuda.device_count()==8'
"$PY" -m unittest v4.test_contract v3.test_geometry -v
mkdir -p "$ROOT/training"
sha256sum v4/*.py > "$ROOT/submitted_code_sha256.txt"
CUDA_VISIBLE_DEVICES=0 timeout 1200 "$PY" -u -m v4.smoke > "$ROOT/wan_smoke.log" 2>&1
for attempt in $(seq 1 120); do
 if test -f "$ROOT/RGB_READY.json"; then break; fi
 sleep 10
done
test -f "$ROOT/RGB_READY.json"
pids=()
for gpu in $(seq 0 7); do
 CUDA_VISIBLE_DEVICES="$gpu" timeout 14400 "$PY" -u -m v4.cache --worker "$gpu" > "$ROOT/cache_$gpu.log" 2>&1 &
 pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
test "$fail" = 0
"$PY" -c 'from v4.common import *;assert all((ROOT/f"CACHE_DONE_{i}.json").exists() for i in range(8));dump(ROOT/"CACHE_READY.json",{"workers":8})'
# Equal architecture, initialization, batch and downstream data for all conditions.
for wave in 0 1; do
 if test "$wave" = 0; then seeds=(17 29); else seeds=(43 71); fi
 pids=();gpu=0
 for seed in "${seeds[@]}"; do
  for condition in human_joint bridge_joint human_video_only robotwin_only; do
   CUDA_VISIBLE_DEVICES="$gpu" timeout 172800 "$PY" -u -m v4.train --condition "$condition" --seed "$seed" --pre-steps 20000 --fine-steps 10000 --micro 2 --accum 8 > "$ROOT/training/seed${seed}_${condition}.log" 2>&1 &
   pids+=("$!");gpu=$((gpu+1))
  done
 done
 fail=0
 for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
 test "$fail" = 0
done
"$PY" -c 'from v4.common import *;assert all((ROOT/f"training/seed{s}/{g}/DONE.json").exists() for s in SEEDS for g in GROUPS);dump(ROOT/"TRAIN_DONE.json",{"models":16})'
# Generate held-out comparisons from predicted actions; these are not rollout success evidence.
pids=();gpu=0
for condition in human_joint bridge_joint human_video_only robotwin_only; do
 CUDA_VISIBLE_DEVICES="$gpu" timeout 3600 "$PY" -u -m v4.visualize_predictions --checkpoint "$ROOT/training/seed17/$condition/finetune_best.pt" --count 4 > "$ROOT/training/visualize_$condition.log" 2>&1 &
 pids+=("$!");gpu=$((gpu+1))
done
fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
test "$fail" = 0
