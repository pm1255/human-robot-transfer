#!/usr/bin/env bash
set -euo pipefail
cd /user/panmiao/human-robot-transfer
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTHONPATH="/user/panmiao/human-robot-transfer/vendor/sam3-main:/user/panmiao/human-robot-transfer"
exec .venv-sam3/bin/python -u -m human2robot.retarget_cloth.segment_human --root outputs/robot_composite_20261020 --objects outputs/object_annotation_20261020 "$@"
