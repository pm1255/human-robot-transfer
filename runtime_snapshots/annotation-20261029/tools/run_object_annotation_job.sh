#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/human-robot-transfer
OUT=$ROOT/outputs/object_annotation_20261020
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
cd "$ROOT"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$ROOT/vendor/sam3-main:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="$ROOT/.venv-sam3/bin/python"
# Validate actual model/session and video encoding before spending a full run.
"$PY" -m human2robot.object_annotation.sam3_video \
 --video "$ROOT/inputs/unlabeled/egoverse_2026-04-28-21-04-25-000000.mp4" \
 --checkpoint /mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt \
 --output "$OUT/preflight" --prompt 'white t-shirt' --chunk-size 16 --max-frames 16
"$PY" -m human2robot.object_annotation.render --output "$OUT/preflight"
"$PY" -m human2robot.object_annotation.sam3_video \
 --video "$ROOT/inputs/unlabeled/egoverse_2026-04-28-21-04-25-000000.mp4" \
 --checkpoint /mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt \
 --output "$OUT" --prompt 'white t-shirt' --chunk-size 120
"$PY" -m human2robot.object_annotation.render --output "$OUT"
"$PY" -m human2robot.object_annotation.refine_missing --output "$OUT"
"$PY" -m human2robot.object_annotation.render --output "$OUT"
# Guided correction prepared after inspecting early false positives.
for retry in $(seq 1 120); do
  test -f "$ROOT/human2robot/object_annotation/guided_redetect.py" && break
  sleep 5
done
"$PY" -m human2robot.object_annotation.guided_redetect --output "$OUT" --guides "$OUT/garment_guides.json" --sample-only
"$PY" -m human2robot.object_annotation.render --output "$OUT"

for retry in $(seq 1 240); do
  test -f "$OUT/guided_full_ready" && break
  sleep 5
done
"$PY" -m human2robot.object_annotation.instance_track --output "$OUT" --guides "$OUT/garment_guides.json" --pilot
"$PY" -m human2robot.object_annotation.render --output "$OUT"

for retry in $(seq 1 180); do
 test -f "$OUT/instance_full_ready" && break
 sleep 5
done
"$PY" -m human2robot.object_annotation.adaptive_segment --output "$OUT" --guides "$OUT/garment_guides.json" --sample-only
"$PY" -m human2robot.object_annotation.render --output "$OUT"

for retry in $(seq 1 180); do
 test -f "$OUT/adaptive_full_ready" && break
 sleep 5
done
"$PY" -m human2robot.object_annotation.adaptive_segment --output "$OUT" --guides "$OUT/garment_guides.json" --sample-only
"$PY" -m human2robot.object_annotation.render --output "$OUT"

for retry in $(seq 1 180); do
 test -f "$OUT/adaptive_v2_ready" && break
 sleep 5
done
"$PY" -m human2robot.object_annotation.adaptive_segment --output "$OUT" --guides "$OUT/garment_guides.json"
"$PY" -m human2robot.object_annotation.render --output "$OUT"
"$PY" -m human2robot.object_annotation.publish --output "$OUT" --web "$OUT/web_publish"
"$PY" -m human2robot.object_annotation.adaptive_segment --output "$OUT" --guides "$OUT/garment_guides.json" --expand --sample-only
for retry in $(seq 1 180); do
 test -f "$OUT/edge_full_ready" && break
 sleep 5
done
"$PY" -m human2robot.object_annotation.adaptive_segment --output "$OUT" --guides "$OUT/garment_guides.json" --expand
"$PY" -m human2robot.object_annotation.render --output "$OUT"
"$PY" -m human2robot.object_annotation.publish --output "$OUT" --web "$OUT/web_publish"
