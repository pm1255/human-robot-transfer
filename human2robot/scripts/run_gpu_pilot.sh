#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/workspace/human2robot-rgb-pilot-20260923
RELEASE="$ROOT/releases/pilot_v2"
OUT="$ROOT/training_gpu_seed17_v2"
mkdir "$OUT"
cd "$RELEASE"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python3 -c 'import torch,sys; print(sys.executable,torch.__version__,torch.version.cuda); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()' | tee "$OUT/environment.log"
sha256sum train_pilot.py | tee "$OUT/code_sha256.txt"
timeout 1800 python3 -u train_pilot.py --human-root "$ROOT/code/outputs/real" \
  --robot-data "$ROOT/data/robotwin_pilot.npz" --out "$OUT" \
  --steps 120 --robot-steps 200 --seed 17 --device cuda | tee "$OUT/train.log"
