#!/bin/bash
set -euo pipefail
cd /user/panmiao/human-robot-transfer
export PYTHONPATH=$PWD:$PWD/vendor/sam3-main:$PWD/vendor/MoGe-main:$PWD/vendor/FlexGEMM-b2fadb29d41846c7981ade6801ffc689fae119cf:$PWD/vendor/utils3d-moge:$PWD/vendor/pipeline
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
.venv-sam3/bin/python tools/diagnose_unified_sam.py > outputs/unified_multimodal_20261020/diagnostic.log 2>&1
.venv-sam3/bin/python -m human2robot.object_annotation.unified_multimodal --root outputs/unified_multimodal_20261020 --stage depth > outputs/unified_multimodal_20261020/depth.log 2>&1
