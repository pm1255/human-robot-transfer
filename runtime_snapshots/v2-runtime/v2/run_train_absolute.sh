#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/workspace/human2robot-v2-20260923
cd "$ROOT/code"
export OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
python3 -c 'import torch; print(torch.__version__,torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
python3 -c 'import json; from pathlib import Path; r=Path("../annotations/human"); assert len(json.loads((r/"manifest.json").read_text()))==96; assert len(list(r.glob("*/supervision.npz")))==96'
mkdir -p "$ROOT/training_absolute"
sha256sum v2/policy.py v2/train.py > "$ROOT/training_absolute/code_sha256.txt"
# Two independent seeds; fixed stage lengths and identical robot minibatch stream within each seed.
for seed in 17 29; do
 timeout 5400 python3 -u -m v2.train --robot "$ROOT/data/robot" --human "$ROOT/annotations/human" --out "$ROOT/training_absolute/seed$seed" --seed "$seed" --pre-steps 2000 --fine-steps 8000 --action-representation absolute | tee "$ROOT/training_absolute/seed$seed.log"
done
