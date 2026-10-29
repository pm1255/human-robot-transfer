#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/human-robot-transfer
OUT=$ROOT/outputs/object_annotation_20261020
cd "$ROOT"
exec > >(tee -a "$OUT/final_segmentation_job.log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT/vendor/sam3-main:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/.venv-sam3/bin/python"
# Source chunks, model provenance and target guides were prepared and inspected.
"$PY" -m human2robot.object_annotation.adaptive_segment --output "$OUT" --guides "$OUT/garment_guides.json" --expand
"$PY" -m human2robot.object_annotation.render --output "$OUT"

"$PY" -m human2robot.object_annotation.publish --output "$OUT" --web "$OUT/web_publish"
export PYTHONPATH="$ROOT:$ROOT/vendor/FoundationPose-main:$ROOT/vendor/FoundationPose-main/mycpp/build"
"$ROOT/.venv-foundation/bin/python" -m human2robot.object_annotation.foundation_preflight --output "$OUT"
for retry in $(seq 1 60); do
 test -f "$ROOT/models/moge-3-vitl/model.pt" && break
 sleep 5
done
if test -f "$ROOT/models/moge-3-vitl/model.pt"; then
 bash "$ROOT/tools/run_cloth_depth_job.sh"
fi
if test -f "$OUT/depth_quality.json" && test -f "$ROOT/vendor/FoundationPose-main/weights/2024-01-11-20-02-45/model_best.pth"; then
 bash "$ROOT/tools/run_cloth_pose_job.sh"
fi
