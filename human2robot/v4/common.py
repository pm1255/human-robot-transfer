import os, json, hashlib
from pathlib import Path

ROOT = Path(os.environ.get('H2R_V4_ROOT', '/user/panmiao/workspace/human2robot-wan-v4-20260924'))
V3 = Path('/user/panmiao/workspace/human2robot-v3-20260923')
WEIGHTS = Path(os.environ.get('WAN_WEIGHTS', '/user/liuhanyu/model_ckpt/Wan2.1-T2V-1.3B'))
VENDOR = Path(__file__).resolve().parents[1] / 'vendor/Wan2.1'
SIZE, FRAMES, HORIZON = 256, 5, 4
EVAL_TEXT = 'Pick up the empty cup and place it on the coaster.'
GROUPS = ['human_joint', 'bridge_joint', 'human_video_only', 'robotwin_only']
SEEDS = [17, 29, 43, 71]

def dump(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)); tmp.replace(path)

def key(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()

def save_tensor(path, value):
    import torch
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
