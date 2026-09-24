#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
command -v ffmpeg >/dev/null
command -v nvidia-smi >/dev/null || { echo 'NVIDIA CUDA GPU required.' >&2; exit 2; }
"${PYTHON_BIN:-python3}" -m venv .venv-vace
PY=.venv-vace/bin/python
"$PY" -m pip install --upgrade pip
# Install a CUDA-compatible PyTorch build first if the default wheel does not suit your driver.
"$PY" -m pip install 'torch>=2.4' 'torchvision>=0.19'
mkdir -p third_party models
if [ ! -d third_party/Wan2.1 ]; then git clone --depth 1 https://github.com/Wan-Video/Wan2.1.git third_party/Wan2.1; fi
# This project's native inference wrapper supports PyTorch SDPA without flash-attn.
"$PY" - <<'PY'
from pathlib import Path
p=Path('third_party/Wan2.1/requirements.txt')
Path('third_party/wan-no-flash.txt').write_text('\n'.join(x for x in p.read_text().splitlines() if not x.strip().replace('-','_').startswith('flash_attn'))+'\n')
PY
"$PY" -m pip install -r third_party/wan-no-flash.txt 'huggingface_hub[cli]' Pillow
"$PY" -c 'import torch; assert torch.cuda.is_available(), "PyTorch cannot access CUDA; install the wheel matching your driver before downloading weights"'
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-VACE-1.3B',local_dir='models/Wan2.1-VACE-1.3B')
PY
printf '%s\n' 'Ready: source .venv-vace/bin/activate' 'Weights downloaded; supply your source video, aligned mask video and human reference image.'
