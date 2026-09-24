#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/workspace/human2robot-v3-20260923
for attempt in $(seq 1 60); do
 if test -f "$ROOT/TRAIN_REAL_READY.json"; then
  exec bash "$ROOT/code/v3/run_real.sh"
 fi
 echo "Waiting for CPU data/model preflight: attempt $attempt/60"
 sleep 10
done
echo 'Preflight not ready after 10 minutes; exit without training.' >&2
exit 1
