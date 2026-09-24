"""Paired scene/seed bootstrap. Refuse to publish a complete score from partial rollouts."""
import argparse,json
from pathlib import Path
import numpy as np
from v3.common import dump

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);seeds=[17,29,43,71];groups=['human','bridge','human_shuffled','robotwin_only'];mat={};scene_ids=None
for group in groups:
    rows=[]
    for seed in seeds:
        episodes=json.loads((root/f'seed{seed}'/group/'summary.json').read_text())['episodes']
        by={int(r['seed']):r for r in episodes}
        assert len(by)==len(episodes) and len(by)>=100, f'Incomplete or duplicated: {group}, {seed}'
        ids=sorted(by)
        if scene_ids is None:scene_ids=ids
        assert ids==scene_ids,'Unpaired scene sets'
        rows.append([float(by[i]['success']) for i in ids])
    mat[group]=np.array(rows)
rng=np.random.default_rng(223);draws={g:[] for g in groups}
for _ in range(5000):
    seed_idx=rng.integers(0,len(seeds),len(seeds));scene_idx=rng.integers(0,len(scene_ids),len(scene_ids))
    for g in groups:draws[g].append(float(mat[g][seed_idx][:,scene_idx].mean()))
report={'scenes':len(scene_ids),'training_seeds':seeds,'bootstrap':'paired crossed resampling of seeds and scenes, 5000 draws','conditions':{},'comparisons':{},'noninferiority_margin':.05}
for g in groups:report['conditions'][g]={'success_rate':float(mat[g].mean()),'per_seed':mat[g].mean(1).tolist(),'ci95':np.quantile(draws[g],[.025,.975]).tolist()}
for g in ['bridge','human_shuffled','robotwin_only']:
    diff=np.array(draws['human'])-draws[g];lo,hi=np.quantile(diff,[.025,.975]);report['comparisons']['human_minus_'+g]={'difference':float(mat['human'].mean()-mat[g].mean()),'ci95':[float(lo),float(hi)]}
report['uncertainty_note']='Only four training seeds; paired bootstrap is exploratory and may give degenerate intervals at all-success/all-failure boundaries.'
report['noninferior_to_bridge_on_this_task']=report['comparisons']['human_minus_bridge']['ci95'][0]>-.05
report['evidence_for_motion_label_benefit']=report['comparisons']['human_minus_human_shuffled']['ci95'][0]>0 and report['comparisons']['human_minus_robotwin_only']['ci95'][0]>0
dump(a.out,report);print(json.dumps(report,indent=2))
