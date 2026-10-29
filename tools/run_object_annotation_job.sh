#!/usr/bin/env bash
set -euo pipefail
ROOT=${H2R_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
OUT=${H2R_OUT:-$ROOT/outputs/object_annotation_20261020}
cd "$ROOT"
if [ -z "${SAM3_CHECKPOINT:-}" ] && [ ! -f "$ROOT/models/sam3.1/sam3.1_multiplex.pt" ] && [ -f /mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt ]; then
  export SAM3_CHECKPOINT=/mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt
fi
exec > >(tee -a "$OUT/final_segmentation_job.log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT/vendor/sam3-main:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY=${H2R_PYTHON:-$ROOT/.venv-sam3/bin/python}
# Source chunks, model provenance and target guides were prepared and inspected.
"$PY" -m human2robot.object_annotation.adaptive_segment --output "$OUT" --guides "$OUT/garment_guides.json" --expand
"$PY" -m human2robot.object_annotation.render --output "$OUT"

"$PY" -m human2robot.object_annotation.publish --output "$OUT" --web "$OUT/web_publish"
