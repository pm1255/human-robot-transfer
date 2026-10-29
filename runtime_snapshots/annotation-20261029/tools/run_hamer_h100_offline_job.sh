#!/usr/bin/env bash
set -euo pipefail
ROOT=/user/panmiao/human-robot-transfer
WORK=/tmp/hamer-runtime
PKG=$WORK/pkgs
WHEELS=$ROOT/models/hamer/wheels
LOG=$ROOT/outputs/hamer_job_offline.log
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

# The batch network is intentionally disabled. Install the pre-staged wheels
# into an ephemeral target directory, leaving the shared source untouched.
rm -rf "$PKG"
mkdir -p "$PKG"
for wheel in "$WHEELS"/*.whl; do
  echo "unpack $(basename "$wheel")"
  unzip -q -o "$wheel" -d "$PKG"
done
mkdir -p "$WORK/chumpy-src"
tar -xzf "$WHEELS/chumpy-0.70.tar.gz" -C "$WORK/chumpy-src"
cp -R "$WORK/chumpy-src"/chumpy-0.70/chumpy "$PKG/"
cat > "$PKG/sitecustomize.py" <<'PY2'
import inspect
if not hasattr(inspect, "getargspec"):
    inspect.getargspec = inspect.getfullargspec
PY2
export PYTHONPATH="$PKG:$ROOT/vendor/hamer:$ROOT:${PYTHONPATH:-}"
python - <<'PY'
mods=['mediapipe','cv2','smplx','pyrender','trimesh','yacs','timm','einops','pytorch_lightning','skimage']
for name in mods:
    m=__import__(name); print('import_ok',name,getattr(m,'__version__',''))
import torch
print('cuda_after_deps',torch.cuda.is_available(),torch.cuda.get_device_name(0))
PY

export HAMER_ROOT="$ROOT/vendor/hamer"
export HAMER_CHECKPOINT="$ROOT/models/hamer/hamer_ckpts/checkpoints/hamer.ckpt"
export HAMER_INPUT="$ROOT/inputs/unlabeled/humanego_serve_bread_000_6s.mp4"
export HUMANEGO_INPUT="$ROOT/inputs/unlabeled/humanego_serve_bread_000_6s.mp4"
export HAND_LANDMARKER_MODEL="$ROOT/models/mediapipe/hand_landmarker.task"
export HAMER_SHARED_ROOT="$ROOT"
export HAMER_OUT_ROOT="$ROOT/outputs/humanego_serve_bread_000_6s"
cd "$ROOT/models/hamer"
export HUMANEGO_HAND_OUT="$ROOT/outputs/humanego_serve_bread_000_6s/hand"
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
