#!/usr/bin/env bash
set -euo pipefail
ROOT=${H2R_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${H2R_OUT:-$ROOT/outputs/unified_30_20261021}
cd "$ROOT"
if [ -z "${SAM3_CHECKPOINT:-}" ] && [ ! -f "$ROOT/models/sam3.1/sam3.1_multiplex.pt" ] && [ -f /mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt ]; then
  export SAM3_CHECKPOINT=/mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT:$ROOT/vendor/sam3-main:$ROOT/vendor/MoGe-main:$ROOT/vendor/FlexGEMM-b2fadb29d41846c7981ade6801ffc689fae119cf:$ROOT/vendor/utils3d-moge:$ROOT/vendor/pipeline"
PY=${H2R_PYTHON:-$ROOT/.venv-sam3/bin/python}
pids=()
for gpu in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES=$gpu "$PY" -m human2robot.object_annotation.unified_multimodal --root "$OUT" --stage depth --config "$OUT/depth_shard_${gpu}.json" > "$OUT/depth_gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
