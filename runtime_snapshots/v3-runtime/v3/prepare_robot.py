"""Bridge realized TCP trajectories and RoboTwin native action_eef, never joint labels."""
import json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from v3.common import *


def episodes(root):
    return pd.concat([pd.read_parquet(p) for p in sorted((root/'meta/episodes').rglob('*.parquet'))])


def video(root,info,row,key,n):
    pre='videos/'+key
    path=root/info['video_path'].format(video_key=key,chunk_index=int(row[pre+'/chunk_index']),file_index=int(row[pre+'/file_index']))
    raw=subprocess.check_output(['ffmpeg','-v','error','-filter_threads','1','-filter_complex_threads','1','-threads','1','-ss',str(row[pre+'/from_timestamp']),'-i',str(path),'-frames:v',str(n),'-vf',f'scale={SIZE}:{SIZE}','-an','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],timeout=120)
    frames=np.frombuffer(raw,np.uint8).reshape(-1,SIZE,SIZE,3)
    if len(frames)!=n:raise ValueError(f'decode mismatch {len(frames)} != {n}')
    return frames


def collect_bridge():
    dest=ROOT/'bridge';dest.mkdir(exist_ok=True)
    info=json.loads((BRIDGE/'meta/info.json').read_text()); assert info['fps']==FPS
    ds=episodes(BRIDGE);ds=ds[ds.tasks.apply(lambda ts:any(category(t) for t in ts))]
    ds=ds.iloc[np.random.default_rng(429).permutation(len(ds))]
    records=[];counts={'train':0,'validation':0,'test':0}; cache_path=None;cache=None
    for _,r in ds.iterrows():
        ep=int(r.episode_index);split=split_episode('bridge',ep)
        if counts[split]>={'train':18000,'validation':1200,'test':1200}[split]:continue
        path=dest/f'bridge_{ep:06d}.npz';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            rec=json.loads(meta.read_text());records.append(rec);counts[split]+=rec['windows'];continue
        try:
            dp=BRIDGE/info['data_path'].format(chunk_index=int(r['data/chunk_index']),file_index=int(r['data/file_index']))
            if dp!=cache_path:cache=pd.read_parquet(dp);cache_path=dp
            d=cache[cache.episode_index==ep].sort_values('frame_index');n=len(d)
            if n<=HORIZON:continue
            assert np.allclose(np.diff(d.timestamp),1/FPS,atol=1e-4)
            assert np.array_equal(d.frame_index,np.arange(n))
            p=np.concatenate([np.stack(d['observation.state.eef_pose']),np.stack(d['observation.state.gripper'])],1).astype(np.float32)
            norms=np.linalg.norm(p[:,3:7],axis=1)
            if not np.isfinite(p).all() or not np.allclose(norms,1,atol=.02):continue
            # Fixed camera; preserve the native 5 Hz timestamps and quaternion xyzw.
            images=video(BRIDGE,info,r,'observation.images.cam_ext_0',n)
            windows=np.array([[t,0] for t in range(n-1)],np.int32)
            np.savez_compressed(path,images=images,poses=p[:,None],windows=windows)
            rec=dict(id=path.stem,episode=ep,file=path.name,frames=n,windows=len(windows),split=split,text=list(r.tasks),category='container_transfer',sha256=sha(path),source=str(dp),target='realized future TCP in current tool frame, NOT joint commands',gripper='measured normalized opening, larger=open')
            dump(meta,rec);records.append(rec);counts[split]+=len(windows)
            print('BRIDGE',ep,counts,flush=True)
        except Exception as e:print('BRIDGE ERROR',ep,repr(e),flush=True)
        if all(counts[s]>=v for s,v in {'train':18000,'validation':1200,'test':1200}.items()):break
    dump(ROOT/'bridge_manifest.json',records)
    assert counts['train']>=10000,counts


def collect_sim():
    dest=ROOT/'sim';dest.mkdir(exist_ok=True);info=json.loads((SIM/'meta/info.json').read_text());ds=episodes(SIM)
    order=np.random.default_rng(429).permutation(sorted(ds.episode_index.unique()))
    split={int(ep):'train' if i<40 else 'validation' if i<45 else 'test' for i,ep in enumerate(order)}
    records=[];cache=None;cache_path=None
    for _,r in ds.sort_values('episode_index').iterrows():
        ep=int(r.episode_index);path=dest/f'sim_{ep:03d}.npz';meta=path.with_suffix('.json')
        if path.exists() and meta.exists():records.append(json.loads(meta.read_text()));continue
        dp=SIM/info['data_path'].format(chunk_index=int(r['data/chunk_index']),file_index=int(r['data/file_index']))
        if dp!=cache_path:cache=pd.read_parquet(dp);cache_path=dp
        d=cache[cache.episode_index==ep].sort_values('frame_index');n=len(d)
        p=np.stack(d['observation.endpose']).reshape(n,2,8).astype(np.float32)
        action=np.stack(d['action_eef']).reshape(n,2,8).astype(np.float32)
        # Upstream RoboTwin wxyz -> canonical xyzw, retaining grip at column 7.
        p=p[:,:,[0,1,2,4,5,6,3,7]];action=action[:,:,[0,1,2,4,5,6,3,7]]
        assert np.isfinite(p).all() and np.isfinite(action).all()
        assert np.allclose(np.linalg.norm(p[:,:,3:7],axis=-1),1,atol=.02)
        images=video(SIM,info,r,'observation.images.head_camera',n)
        windows=np.array([[t,s] for t in range(n-HORIZON) for s in range(2)],np.int32)
        np.savez_compressed(path,images=images,poses=p,actions=action,windows=windows)
        rec=dict(id=path.stem,episode=ep,file=path.name,frames=n,windows=len(windows),split=split[ep],sha256=sha(path),source=str(dp),fps=info['fps'],task='place_empty_cup',target='native absolute action_eef converted to current-TCP relative targets',joint_labels_consumed=False)
        dump(meta,rec);records.append(rec);print('SIM',ep,n,flush=True)
    dump(ROOT/'sim_manifest.json',records)

if __name__=='__main__':
    collect_sim(); collect_bridge()
