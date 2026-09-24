"""Freeze exact unique windows, provenance, and episode-disjoint partitions."""
import json
import numpy as np
from v3.common import *


def build(domain):
    folder='human_stable' if domain=='human' else domain
    recs=json.loads((ROOT/f'{folder}_manifest.json').read_text())
    rows={s:[] for s in ['train','validation','test']}
    source_seen=set()
    for r in recs:
        if 'error' in r or not r['windows']:continue
        if domain=='human' and r.get('algorithm_version')!=5:continue
        if domain=='human':assert r['weak_pretrain_eligible'] and not r['robot_training_valid'] and not r['hardware_labels_consumed']
        if domain=='bridge' and not any(category(t) for t in r['text']):continue
        p=ROOT/folder/r['file']; d=np.load(p); w=d['windows']
        for wi,(t,side) in enumerate(w):
            if domain=='human' and not d['window_masks'][wi,0]:continue
            if domain=='human':
                native_fps=30 if r['episode']>=1000000 else 59.94005994
                key=(r['source'],round((r['start']+int(t)/FPS)*native_fps),int(side))
                if key in source_seen:continue
                source_seen.add(key)
            rows[r['split']].append([r['file'],int(t),int(side),r['episode']])
    rng=np.random.default_rng(429)
    for s in rows:
        rng.shuffle(rows[s])
        # At most 160 windows from one source episode; no duplicated start/hand.
        counts={};keep=[]
        for row in rows[s]:
            ep=row[3]
            if counts.get(ep,0)>=160 and domain!='sim':continue
            counts[ep]=counts.get(ep,0)+1;keep.append(row)
        rows[s]=keep
    if domain!='sim':
        assert len(rows['train'])>=10000, f'{domain}: only {len(rows["train"])} distinct valid training windows; refusing to repeat windows to claim 10k'
        rows['train']=rows['train'][:10000]
        for s in ['validation','test']:rows[s]=rows[s][:1000]
    assert all(len(rows[s])>0 for s in rows)
    sets={s:{r[3] for r in rows[s]} for s in rows}
    assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
    for s in rows:assert len({tuple(r[:3]) for r in rows[s]})==len(rows[s])
    dump(ROOT/f'{domain}_index.json',rows)
    return {s:{'windows':len(rows[s]),'episodes':len(sets[s]),'unique_anchor_frames':len({(r[0],r[1]) for r in rows[s]})} for s in rows}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--domains',nargs='+',default=['human','bridge','sim']);a=p.parse_args()
    audit={d:build(d) for d in a.domains}
    audit.update(human_hardware_labels_consumed=False,source_pose_claim='human weak RGB pseudo TCP; robot observed TCP',human_original_session_split_verified=False,pretrain_fps=FPS,pretrain_supervised_horizon=1,finetune_horizon=HORIZON,sim_fps=15,split_seed=429)
    name='DATA_READY.json' if 'human' in a.domains else 'DATA_READY_REAL.json'
    dump(ROOT/name,audit);print(json.dumps(audit,indent=2),flush=True)
