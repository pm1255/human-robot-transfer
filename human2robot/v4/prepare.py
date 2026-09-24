"""Freeze task text + true RGB clips + gripper targets; never repeat samples to reach 10k."""
import argparse, collections, concurrent.futures, json, subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from v4.common import *
from v3.common import local_targets, state7, BRIDGE, SIM

def metadata(root):
    info=json.loads((root/'meta/info.json').read_text())
    df=pd.concat([pd.read_parquet(p) for p in sorted((root/'meta/episodes').rglob('*.parquet'))])
    return info,{int(r.episode_index):r for _,r in df.iterrows()}

def instruction(value):
    if isinstance(value,str): out=value.strip()
    else: out=next((str(x).strip() for x in value if str(x).strip()),'')
    if not out: raise ValueError('Missing language instruction; do not silently replace with generic text')
    return out

def freeze():
    ROOT.mkdir(parents=True,exist_ok=True); result={}; episode_jobs={}
    sim_info,sim_meta=metadata(SIM); bridge_info,bridge_meta=metadata(BRIDGE)
    for domain,folder in [('human','human_stable'),('bridge','bridge'),('sim','sim')]:
        records=json.loads((V3/(folder+'_manifest.json')).read_text()); groups=collections.defaultdict(list)
        for r in records:
            if r.get('error') or (domain=='human' and r.get('algorithm_version')!=5): continue
            if r['frames'] <= (12 if domain=='sim' else 4): continue
            if domain=='human':
                text=instruction(r['text']); provenance=r['annotation_source']
                source=r['source']; start=r['start']; native_fps=5
            else:
                base,info,meta=(SIM,sim_info,sim_meta) if domain=='sim' else (BRIDGE,bridge_info,bridge_meta)
                rr=meta[r['episode']]; text=instruction(rr.tasks); provenance='dataset_episode_tasks'
                vk='observation.images.head_camera' if domain=='sim' else 'observation.images.cam_ext_0'
                pre='videos/'+vk
                source=str(base/info['video_path'].format(video_key=vk,chunk_index=int(rr[pre+'/chunk_index']),file_index=int(rr[pre+'/file_index'])))
                start=float(rr[pre+'/from_timestamp']); native_fps=info['fps']
            stride=3 if domain=='sim' else 1
            episode_key=domain+'_'+r['id']; epjob=dict(source=source,start=start,n=r['frames'],fps=native_fps)
            with np.load(V3/folder/r['file']) as d:
                poses=d['poses']; win=d['windows']; actions=d['actions'] if 'actions' in d else None
                for t,s in win.tolist():
                    if t+4*stride>=r['frames']: continue
                    target_poses=poses[t+np.arange(1,5)*stride,s] if actions is None else actions[t:t+4,s]
                    mask=np.ones(4,np.float32)
                    if domain=='human':
                        for j in range(1,5):
                            ok=d['observed'][t:t+j+1,s].all()
                            q=poses[t:t+j+1,s]
                            from scipy.spatial.transform import Rotation
                            dr=(Rotation.from_quat(q[:-1,3:7]).inv()*Rotation.from_quat(q[1:,3:7])).magnitude()
                            ok=ok and (np.linalg.norm(np.diff(q[:,:3],axis=0),axis=1)<.15).all() and (dr<1.2).all()
                            ok=ok and (d['background_flow'][t+1:t+j+1]<.045).all()
                            # Confidence weights are weak-label quality, not probability of correctness.
                            err=float(d['reprojection'][t:t+j+1,s].max())
                            mask[j-1]=max(.1,1-err/.035) if ok else 0
                        if not mask[0]: continue
                    uid=key(f'{source}:{start+t/native_fps:.6f}:{s}')
                    groups[r['split']].append(dict(uid=uid,episode=episode_key,t=t,side=s,stride=stride,
                        text=text,text_key=key(text),text_provenance=provenance,source=source,
                        timestamp=start+t/native_fps,state=state7(poses[t,s]).tolist(),
                        target=local_targets(poses[t,s],target_poses).tolist(),mask=mask.tolist(),
                        action_dt=(1/15 if domain=='sim' else .2),video_dt=.2,
                        weak_action=(domain=='human')))
            episode_jobs[episode_key]=epjob
        result[domain]={}
        for split,rows in groups.items():
            # Same deterministic freeze for both human ablations; episode-level cap.
            rows=sorted(rows,key=lambda r:key('429:'+r['uid'])); seen=set(); counts=collections.Counter(); kept=[]
            limit={'train':10000,'validation':1000,'test':1000}[split] if domain!='sim' else None
            for row in rows:
                if row['uid'] in seen or (domain!='sim' and counts[row['episode']]>=160): continue
                seen.add(row['uid']); counts[row['episode']]+=1; kept.append(row)
                if limit and len(kept)==limit: break
            if limit and len(kept)!=limit: raise RuntimeError(f'{domain}/{split}: only {len(kept)}/{limit} eligible clips')
            result[domain][split]=kept
        # Video file grouping retained from v3; no cross-split original-source overlap for human.
        if domain=='human':
            sources={s:{r['source'] for r in rows} for s,rows in result[domain].items()}
            assert not sources['train'] & (sources['test']|sources['validation'])
        dump(ROOT/'manifests'/f'{domain}.json',result[domain])
    used={r['episode'] for ds in result.values() for rows in ds.values() for r in rows}
    dump(ROOT/'episode_jobs.json',{k:v for k,v in episode_jobs.items() if k in used})
    texts={r['text_key']:r['text'] for ds in result.values() for rows in ds.values() for r in rows}
    texts[key(EVAL_TEXT)]=EVAL_TEXT
    dump(ROOT/'texts.json',texts)
    audit={d:{s:{'windows':len(rows),'episodes':len({r['episode'] for r in rows}),
                  'unique_instructions':len({r['text'] for r in rows})} for s,rows in splits.items()} for d,splits in result.items()}
    dump(ROOT/'DATA_AUDIT.json',dict(data=audit,raw_video_resolution=SIZE,frames=FRAMES,
        human_text_manually_verified=False,original_human_session_split_verified=False,
        hardware_human_actions_used=False,sim_train_episodes=40,
        notes='5 actual RGB frames at 5Hz. Sim action head remains native 15Hz. Human actions are weak labels.'))
    print(json.dumps(audit),flush=True)

def decode(item):
    name,r=item; dest=ROOT/'rgb'/f'{name}.npy';dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        a=np.load(dest,mmap_mode='r')
        if a.shape==(r['n'],SIZE,SIZE,3): return name
        raise RuntimeError('Invalid RGB cache '+str(dest))
    raw=subprocess.check_output(['ffmpeg','-v','error','-filter_threads','1','-threads','1',
        '-ss',str(r['start']),'-i',r['source'],'-vf',f'fps={r["fps"]},scale={SIZE}:{SIZE}',
        '-frames:v',str(r['n']),'-an','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],timeout=180)
    arr=np.frombuffer(raw,np.uint8).reshape(-1,SIZE,SIZE,3)
    if len(arr)!=r['n']: raise ValueError(f'{name}: raw frame count {len(arr)} != {r["n"]}')
    temp=dest.with_suffix('.tmp.npy');np.save(temp,arr);temp.replace(dest);return name

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--decode-only',action='store_true');p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if not a.decode_only: freeze()
    jobs=json.loads((ROOT/'episode_jobs.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,name in enumerate(pool.map(decode,jobs.items())):
            if i%50==0: dump(ROOT/'PREP_PROGRESS.json',dict(decoded=i+1,total=len(jobs))); print('RGB',i+1,len(jobs),name,flush=True)
    dump(ROOT/'RGB_READY.json',dict(episodes=len(jobs)))
