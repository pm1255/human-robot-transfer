#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/workspace/human2robot-wan-v4-20260924
cd "$ROOT/code"
export PYTHONPATH="$ROOT/deps:$ROOT/code" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
PY=/user/panmiao/workspace/envs/windtunnel/bin/python
"$PY" -c 'import torch,sys;print(sys.executable,torch.__version__,torch.cuda.device_count());assert torch.cuda.device_count()==8'
CUDA_VISIBLE_DEVICES=0 timeout 1200 "$PY" -u -m v4.smoke > "$ROOT/wan_smoke.log" 2>&1
