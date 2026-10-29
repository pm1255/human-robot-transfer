#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/human-robot-transfer
OUT=$ROOT/outputs/object_annotation_20261020
exec > >(tee -a "$OUT/pose_job_fp32.log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="$ROOT:$ROOT/vendor/FoundationPose-main:$ROOT/vendor/FoundationPose-main/mycpp/build"
cd "$ROOT/vendor/FoundationPose-main"
PY="$ROOT/.venv-foundation/bin/python"
PILOT=$OUT/pose_pilot_ieee
mkdir -p "$PILOT/events"
ln -sfn "$OUT/objects.json" "$PILOT/objects.json"
ln -sfn "$OUT/masks" "$PILOT/masks"
ln -sfn "$OUT/pose_pilot/depth" "$PILOT/depth"
ln -sfn "$OUT/pose_pilot/depth_manifest.json" "$PILOT/depth_manifest.json"
ln -sfn "$PILOT/events/pose6d" "$OUT/events/pilot_pose6d_fp32"
"$PY" -m human2robot.object_annotation.foundation_cloth --output "$PILOT" --reference-frame 30 --reference-review "$OUT/pose_pilot/reference_review.json" --max-frames 180
"$PY" -m human2robot.object_annotation.render_pose_pilot --output "$PILOT"
"$PY" -m human2robot.object_annotation.foundation_cloth --output "$OUT" --reference-frame 30 --reference-review "$OUT/pose_pilot/reference_review.json" --wait-depth-seconds 3600
"$PY" -m human2robot.object_annotation.render_pose_pilot --output "$OUT"
"$PY" -m human2robot.object_annotation.validate_geometry_outputs --output "$OUT"
