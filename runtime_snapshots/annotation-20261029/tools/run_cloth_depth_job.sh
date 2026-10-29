#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/human-robot-transfer
OUT=$ROOT/outputs/object_annotation_20261020
cd "$ROOT"
exec > >(tee -a "$OUT/depth_job.log") 2>&1
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT:$ROOT/vendor/MoGe-main:$ROOT/vendor/FlexGEMM-b2fadb29d41846c7981ade6801ffc689fae119cf:$ROOT/vendor/utils3d-moge:$ROOT/vendor/pipeline"
PY="$ROOT/.venv-sam3/bin/python"
"$PY" tools/prepare_flexgemm_compat.py "$ROOT/vendor/FlexGEMM-b2fadb29d41846c7981ade6801ffc689fae119cf"
status() {
 "$PY" -m human2robot.object_annotation.pipeline_status --output "$1" --stage "$2" --status "$3" --reason "$4" --total "$5"
}
stage=depth_preflight
failed() {
 status "$OUT" depth failed "Pipeline failed during $stage; see depth_job.log" 4834
}
trap failed ERR
status "$OUT" depth waiting "Waiting for depth and pose pilot" 4834
status "$OUT/pose_pilot" pose6d waiting "Waiting for pilot depth" 180
ln -sfn "$OUT/depth_preflight/events/depth" "$OUT/events/depth_preflight"
"$PY" -m human2robot.object_annotation.depth_video \
 --video "$ROOT/inputs/unlabeled/egoverse_2026-04-28-21-04-25-000000.mp4" \
 --checkpoint "$ROOT/models/moge-3-vitl/model.pt" --output "$OUT/depth_preflight" --max-frames 2
stage=pilot_depth
"$PY" -m human2robot.object_annotation.depth_video \
 --video "$ROOT/inputs/unlabeled/egoverse_2026-04-28-21-04-25-000000.mp4" \
 --checkpoint "$ROOT/models/moge-3-vitl/model.pt" --output "$OUT/pose_pilot" --max-frames 180
if test -f "$ROOT/vendor/FoundationPose-main/weights/2024-01-11-20-02-45/model_best.pth"; then
 stage=pilot_pose6d
 (
  cd "$ROOT/vendor/FoundationPose-main"
  export PYTHONPATH="$ROOT:$ROOT/vendor/FoundationPose-main:$ROOT/vendor/FoundationPose-main/mycpp/build"
  "$ROOT/.venv-foundation/bin/python" -m human2robot.object_annotation.foundation_cloth --output "$OUT/pose_pilot" --reference-frame 30 --reference-review "$OUT/pose_pilot/reference_review.json" --max-frames 180
 )
fi
stage=full_depth
"$PY" -m human2robot.object_annotation.depth_video \
 --video "$ROOT/inputs/unlabeled/egoverse_2026-04-28-21-04-25-000000.mp4" \
 --checkpoint "$ROOT/models/moge-3-vitl/model.pt" --output "$OUT"
