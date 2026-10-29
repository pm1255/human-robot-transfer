#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/human-robot-transfer
WORK=/tmp/hamer-runtime
ENV=$WORK/venv
LOG=$ROOT/outputs/hamer_job.log
mkdir -p "$WORK" "$ROOT/outputs"
exec > >(tee -a "$LOG") 2>&1

echo "[job] $(date -Is) host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python --version
python - <<'PY'
import torch
print('torch',torch.__version__,'cuda',torch.version.cuda,'available',torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit('CUDA is not available')
print('device',torch.cuda.get_device_name(0))
PY

# Keep the base image intact and install only the inference dependencies in the
# ephemeral job filesystem. The model weights and outputs remain on /user.
python -m venv --system-site-packages "$ENV" || true
source "$ENV/bin/activate"
python -m pip install --disable-pip-version-check --no-cache-dir --upgrade \
  mediapipe==0.10.21 smplx==0.1.28 pyrender==0.1.45 trimesh==4.4.9 \
  yacs==0.1.8 timm==1.0.15 einops==0.8.0 pytorch-lightning==2.5.5 \
  scikit-image==0.24.0 chumpy==0.70 2>&1 | tee -a "$LOG"

export PYTHONPATH="$ROOT/vendor/hamer:$ROOT:${PYTHONPATH:-}"
export HAMER_ROOT="$ROOT/vendor/hamer"
export HAMER_CHECKPOINT="$ROOT/models/hamer/hamer_ckpts/checkpoints/hamer.ckpt"
export HAMER_INPUT="$ROOT/inputs/unlabeled/humanego_serve_bread_000_6s.mp4"
export HAND_LANDMARKER_MODEL="$ROOT/models/mediapipe/hand_landmarker.task"
export HAMER_SHARED_ROOT="$ROOT"
export HAMER_OUT_ROOT="$ROOT/outputs/humanego_serve_bread_000_6s"

# HaMeR resolves MANO paths relative to _DATA, so run from the model cache.
cd "$ROOT/models/hamer"
python "$ROOT/human2robot/v2/run_mediapipe_hand.py"
export HUMANEGO_HAND_JSON="$ROOT/outputs/humanego_serve_bread_000_6s/hand/humanego_serve_bread_000_6s_hand.json"
export HAMER_OUT_DIR="$ROOT/outputs/humanego_serve_bread_000_6s/hamer"
python "$ROOT/human2robot/v2/run_hamer_mesh.py"

OUT_JSON="$HAMER_OUT_DIR/humanego_serve_bread_000_6s_hamer_mesh.json"
OUT_VIDEO="$HAMER_OUT_DIR/humanego_serve_bread_000_6s_hamer_mesh.mp4"
python - "$OUT_JSON" "$OUT_VIDEO" <<'PY'
import json,sys,os
p=json.load(open(sys.argv[1]))
assert p.get('model')=='HaMeR',p.get('model')
assert p.get('vertices_per_hand')==778,p.get('vertices_per_hand')
assert p.get('mesh_frames',0)>0,p.get('mesh_frames')
assert os.path.getsize(sys.argv[2])>0
print('[verified]',json.dumps({k:p[k] for k in ('model','num_frames','mesh_frames','vertices_per_hand')}))
PY
echo "[job] COMPLETE $(date -Is)"
