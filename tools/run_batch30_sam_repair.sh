#!/usr/bin/env bash
set -euo pipefail
ROOT=${H2R_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
BASE=${H2R_OUT:-$ROOT/outputs/unified_30_20261021}
export H2R_BATCH_ROOT="$BASE"
OUT=$BASE/repair_run
cd "$ROOT"
if [ -z "${SAM3_CHECKPOINT:-}" ] && [ ! -f "$ROOT/models/sam3.1/sam3.1_multiplex.pt" ] && [ -f /mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt ]; then
  export SAM3_CHECKPOINT=/mnt/uds-assets-zw/models/sam3.1/sam3.1_multiplex.pt
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONPATH="$ROOT:$ROOT/vendor/sam3-main:$ROOT/vendor/MoGe-main:$ROOT/vendor/FlexGEMM-b2fadb29d41846c7981ade6801ffc689fae119cf:$ROOT/vendor/utils3d-moge:$ROOT/vendor/pipeline"
PY=${H2R_PYTHON:-$ROOT/.venv-sam3/bin/python}
"$PY" - <<'PY'
import pathlib,json,os
r=pathlib.Path(os.environ['H2R_BATCH_ROOT'])
for sid,classes in json.loads((r/'repair_classes.json').read_text()).items():
 source=r/'samples'/sid;dest=r/'repair_run/samples'/sid;out=dest/'multimodal';(out/'classes').mkdir(parents=True,exist_ok=True)
 for name in ['cam_0.mp4','segments.json']:
  if not(dest/name).exists():(dest/name).symlink_to(source/name)
 for name in ['rgb','depth_model','depth.mp4','depth_validation.json']:
  if not(out/name).exists():(out/name).symlink_to(source/'multimodal'/name)
 for cls in (source/'multimodal/classes').iterdir():
  if cls.name not in classes:
   assert(cls/'rows.json').exists()
   if not(out/'classes'/cls.name).exists():(out/'classes'/cls.name).symlink_to(cls)
 (out/'repair_provenance.json').write_text(json.dumps(dict(initial_output=str(source/'multimodal'),corrected_classes=classes,reason='Observed wrong target or lost visible target; new prompt from decoded source frame',verified_accuracy=False),indent=2))
PY
pids=()
for gpu in 0 1 2 3; do
 (
 export CUDA_VISIBLE_DEVICES=$gpu
 "$PY" -m human2robot.object_annotation.unified_multimodal --root "$OUT" --stage sam --config "$BASE/repair_shard_${gpu}.json"
 "$PY" -m human2robot.object_annotation.unified_multimodal --root "$OUT" --stage render --config "$BASE/repair_shard_${gpu}.json"
 ) > "$BASE/repair_gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
