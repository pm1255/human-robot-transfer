#!/usr/bin/env bash
set -euo pipefail
ROOT=${H2R_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${H2R_OUT:-$ROOT/outputs/unified_multimodal_20261020}
cd "$ROOT"
if [ -z "${SAM3_CHECKPOINT:-}" ] && [ ! -f "$ROOT/models/sam3.1/sam3.1_multiplex.pt" ] && [ -f /mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt ]; then
  export SAM3_CHECKPOINT=/mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT:$ROOT/vendor/sam3-main:$ROOT/vendor/MoGe-main:$ROOT/vendor/FlexGEMM-b2fadb29d41846c7981ade6801ffc689fae119cf:$ROOT/vendor/utils3d-moge:$ROOT/vendor/pipeline"
PY=${H2R_PYTHON:-$ROOT/.venv-sam3/bin/python}
"$PY" tools/prepare_flexgemm_compat.py "$ROOT/vendor/FlexGEMM-b2fadb29d41846c7981ade6801ffc689fae119cf"
"$PY" -m human2robot.object_annotation.unified_multimodal --root "$OUT" --stage preflight --sample umi_hifi_pot
"$PY" -m human2robot.object_annotation.unified_multimodal --root "$OUT" --stage sam
"$PY" -m human2robot.object_annotation.unified_multimodal --root "$OUT" --stage render

