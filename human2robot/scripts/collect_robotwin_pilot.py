"""Read-only, bounded RoboTwin LeRobot-v3 subset; preserves stored joint targets."""
import argparse,json,subprocess,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def collect(root,out):
    root=Path(root);out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    info=json.loads((root/'meta/info.json').read_text());fps=info['fps'];key='observation.images.head_camera'
    assert info['features']['action']['shape']==[14]
    assert info['features']['observation.state']['shape']==[14]
    episodes=pd.concat([pd.read_parquet(p) for p in sorted((root/'meta/episodes').glob('*/*.parquet'))])
    selected=list(range(10));arrays={k:[] for k in ['images','previous_images','state','action','episode_id','split','frame_index']};sources=[]
    for ep in selected:
        row=episodes[episodes.episode_index==ep].iloc[0];prefix='videos/'+key
        parquet=root/info['data_path'].format(chunk_index=int(row['data/chunk_index']),file_index=int(row['data/file_index']))
        video=root/info['video_path'].format(video_key=key,chunk_index=int(row[prefix+'/chunk_index']),file_index=int(row[prefix+'/file_index']))
        d=pd.read_parquet(parquet);d=d[d.episode_index==ep].sort_values('frame_index');n=len(d)
        if not np.array_equal(d.frame_index.to_numpy(),np.arange(n)):raise ValueError('Non-contiguous source frames')
        if not np.allclose(np.diff(d.timestamp.to_numpy()),1/fps,atol=1e-4):raise ValueError('Unexpected timebase')
        start=float(row[prefix+'/from_timestamp'])
        raw=subprocess.check_output(['ffmpeg','-v','error','-ss',str(start),'-i',str(video),'-frames:v',str(n),'-vf','scale=96:96','-pix_fmt','rgb24','-f','rawvideo','pipe:1'])
        frames=np.frombuffer(raw,np.uint8).reshape(-1,96,96,3)
        if len(frames)!=n:raise ValueError(f'Frame mismatch {ep}: {len(frames)} != {n}')
        ids=np.arange(1,n,2);states=np.stack(d['observation.state']);actions=np.stack(d['action'])
        if not np.isfinite(states).all() or not np.isfinite(actions).all():raise ValueError('Non-finite robot values')
        for k,v in {'images':frames[ids],'previous_images':frames[ids-1],'state':states[ids],'action':actions[ids],
                    'episode_id':np.full(len(ids),ep),'split':np.full(len(ids),'train' if ep<8 else 'validation'),
                    'frame_index':ids}.items():arrays[k].append(v)
        sources.append({'episode':ep,'source_video':str(video),'start_timestamp':start,'frames':n,'selected_frames':len(ids),
                        'source_parquet':str(parquet),'split':'train' if ep<8 else 'validation'})
        print('collected',ep,len(ids),flush=True)
    np.savez_compressed(out,**{k:np.concatenate(v) for k,v in arrays.items()})
    report={'dataset_root':str(root),'task':'adjust_bottle','source_kind':'robotwin_simulation','source_fps':fps,
        'observation':'previous and current head-camera RGB, current stored 14D joint state',
        'action':'stored action field, unchanged; joint targets plus source gripper values; NOT action_eef',
        'action_names':info['features']['action']['names'],'unit_status':'joint radians expected from RoboTwin; gripper native units, not converted to metres',
        'video_alignment':'episode video from_timestamp + contiguous 15fps frame_index; decoded RGB24 using ffmpeg',
        'sample_stride':2,'split':'8 training episodes / 2 held-out episodes; no frame-level random split',
        'sources':sources,'npz_sha256':hashlib.sha256(out.read_bytes()).hexdigest()}
    out.with_suffix('.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();collect(a.root,a.out)
