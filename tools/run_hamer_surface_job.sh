#!/usr/bin/env bash
set -euo pipefail
ROOT=${HAMER_SHARED_ROOT:-/user/panmiao/human-robot-transfer}
OUT=${HAMER_SURFACE_OUT:-$ROOT/outputs/hand_mesh_repair_20261020/inference_v2}
WORK=$(mktemp -d /tmp/hamer-surface.XXXXXX)
PKG=$WORK/pkgs
WHEELS=$ROOT/models/hamer/wheels
mkdir -p "$PKG" "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
for wheel in "$WHEELS"/*.whl; do
  [[ $(basename "$wheel") == ._* ]] && continue
  unzip -q -o "$wheel" -d "$PKG"
done
tar -xzf "$WHEELS/chumpy-0.70.tar.gz" -C "$WORK"
cp -R "$WORK/chumpy-0.70/chumpy" "$PKG/"
cat > "$PKG/sitecustomize.py" <<'PY'
import inspect
if not hasattr(inspect,'getargspec'): inspect.getargspec=inspect.getfullargspec
PY
export PYTHONPATH="$PKG:$ROOT/vendor/hamer:$ROOT:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA GPU required for this scheduled inference job'
print('GPU',torch.cuda.get_device_name(0),flush=True)
PY
cd "$ROOT/models/hamer"
python -m human2robot.hand_mesh.identity \
  --input "$ROOT/outputs/egoverse_2026-04-28-21-04-25-000000/hand/egoverse_2026-04-28-21-04-25-000000_hand.json" \
  --output "$OUT/hand_identity_repaired.json"
python -m human2robot.hand_mesh.infer \
  --video "$ROOT/inputs/unlabeled/egoverse_2026-04-28-21-04-25-000000.mp4" \
  --hands "$OUT/hand_identity_repaired.json" \
  --checkpoint "$ROOT/models/hamer/hamer_ckpts/checkpoints/hamer.ckpt" \
  --hamer-root "$ROOT/vendor/hamer" --out "$OUT" --batch-size 16
