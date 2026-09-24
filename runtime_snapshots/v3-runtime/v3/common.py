"""V3 gripper-only protocol: RGB + inferred/observed TCP, no arm joints."""
import hashlib, json, re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path('/user/panmiao/workspace/human2robot-v3-20260923')
HUMAN = Path('/user/zhangxueqian/dataset/lerobot_v3.0_v0/ego/EpicKitchens/epic_kitchens_action_clips_lda_full_v3')
BRIDGE = Path('/user/zhangxueqian/dataset/lerobot_v3.0_v1_origin/bridge_orig/chunk-000')
SIM = Path('/user/xuwang/dataset_sft_gop2/robotwin-clean-xvla-action-eef/place_empty_cup')
HORIZON = 4
FPS = 5
SIZE = 128


def dump(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def category(text):
    t=text.lower().strip()
    if not re.search(r'^(pick|take|put|place|move|lift|set|transfer|retrieve|grab|grasp|remove|carry)',t): return None
    t=re.split(r'\b(?:into|onto|from|inside|out of|off|over|under|near|in|on|to)\b',t,maxsplit=1)[0]
    if re.search(r'\b(bag|cloth|towel|packet|wrapper|peel|lid|door|drawer|pour|wash|rinse|stir|cut|slice|cap|faucet)\b',t): return None
    # Restrict to rigid containers; exclude tool/food transfer from this experiment.
    if re.search(r'\b(cup|mug|glass|bottle|bowl|can|jar|pot|pan|plate|container|tray|kettle)\b',t): return 'container_transfer'
    return None


def local_targets(pose, future):
    """Pose is xyz, xyzw, opening. Delta in CURRENT TCP axes; opening absolute."""
    r=Rotation.from_quat(pose[3:7]); rr=Rotation.from_quat(future[...,3:7])
    dp=r.inv().apply(future[...,:3]-pose[:3])
    dr=(r.inv()*rr).as_rotvec()
    return np.concatenate([dp,dr,future[...,7:8]],axis=-1).astype(np.float32)


def state7(pose):
    return np.concatenate([pose[...,:3],Rotation.from_quat(pose[...,3:7]).as_rotvec(),pose[...,7:8]],-1).astype(np.float32)


def apply_target(pose, action):
    r=Rotation.from_quat(pose[3:7]); a=np.asarray(action)
    return np.concatenate([pose[:3]+r.apply(a[:3]),(r*Rotation.from_rotvec(a[3:6])).as_quat(),[np.clip(a[6],0,1)]]).astype(np.float32)


def split_episode(source, episode):
    # Source video is a group. Original EPIC participant/session IDs are absent upstream.
    k=int(hashlib.sha256(f'v3-429:{source}:{episode}'.encode()).hexdigest()[:8],16)%10
    return 'test' if k==0 else 'validation' if k==1 else 'train'
