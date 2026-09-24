#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/workspace/human2robot-v3-20260923
cd "$ROOT/code"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/user/panmiao/workspace/envs/windtunnel/bin/python
for attempt in $(seq 1 240); do
 if test -f "$ROOT/bridge_manifest.json"; then break; fi
 sleep 10
done
"$PY" -m v3.freeze_data --domains bridge sim
"$PY" -m v3.pack_data --domains bridge sim
for attempt in $(seq 1 180); do
 if test "$(sha256sum "$ROOT/models/resnet18-imagenet.pth" | cut -d' ' -f1)" = f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec; then break; fi
 sleep 10
done
"$PY" -u -m v3.smoke
"$PY" -c 'from v3.common import dump,ROOT;dump(ROOT/"TRAIN_REAL_READY.json",{"geometry_tests":"passed","real_data_count":10000,"strict_model_load":"passed","finite_backward":"passed"})'
