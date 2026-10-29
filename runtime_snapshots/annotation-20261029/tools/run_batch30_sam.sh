#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/human-robot-transfer
OUT=$ROOT/outputs/unified_30_20261021
cd "$ROOT"
exec > >(tee -a "$OUT/sam_job.log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT:$ROOT/vendor/sam3-main:$ROOT/vendor/MoGe-main:$ROOT/vendor/FlexGEMM-b2fadb29d41846c7981ade6801ffc689fae119cf:$ROOT/vendor/utils3d-moge:$ROOT/vendor/pipeline"
PY="$ROOT/.venv-sam3/bin/python"
"$PY" -m human2robot.object_annotation.unified_multimodal --root "$OUT" --stage sam --config "$OUT/sam_config.json"
"$PY" -m human2robot.object_annotation.unified_multimodal --root "$OUT" --stage render --config "$OUT/sam_config.json"
