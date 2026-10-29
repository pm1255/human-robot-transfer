"""Audit encoded temporal chunks, including their output concatenation joins."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from human2robot.object_annotation.common import atomic_json, Progress


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    before='outputs/robot_contact_20261020/background_full/robot_source_composite.mp4'
    parts=sorted((a.root/'full').glob('part_*'))
    progress=Progress('outputs/object_annotation_20261020','robot_temporal_full_encoded_audit',4834)
    def audit(part):
        start=int(part.name.split('_')[1]);out=a.root/'encoded_audit'/part.name;out.mkdir(parents=True,exist_ok=True)
        with (out/'audit.log').open('w') as log:
            subprocess.run([sys.executable,'-m','human2robot.retarget_cloth.audit_temporal_video','--before',before,'--after',str(part/'robot_source_composite.mp4'),'--root',str(out),'--start',str(start)],stdout=log,stderr=subprocess.STDOUT,check=True)
        return json.loads((out/'encoded_temporal_audit.json').read_text())
    results=[];done=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(audit,parts):
            results.append(result);done+=result['summary']['frames'];progress.update(done,phase='encoded_video_temporal_comparison')
    # Two-frame windows assess the concatenation boundaries, which separate
    # chunk metrics cannot see. Warm-up frames are never included in outputs.
    joins=[]
    for i in [600,1200,1800,2400,3000,3600,4200]:
        out=a.root/'encoded_audit'/f'join_{i:06d}';out.mkdir(parents=True,exist_ok=True)
        with (out/'audit.log').open('w') as log:
            subprocess.run([sys.executable,'-m','human2robot.retarget_cloth.audit_temporal_video','--before',before,'--after',str(a.root/'full/robot_source_composite.mp4'),'--root',str(out),'--start',str(i-1),'--after-start',str(i-1),'--count','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
        joins.append(json.loads((out/'encoded_temporal_audit.json').read_text()))
    records=[r for d in results for r in d['frames']]
    records.extend(d['frames'][-1] for d in joins)
    scored=[r for r in records if r['scored']];n=sum(r['pixels'] for r in scored)
    metrics={s:sum(r['pixels']*r['errors'][s]['mean_rgb_difference'] for r in scored)/max(n,1) for s in ['before','after']}
    summary=dict(total_frames=4834,scored_transitions=len(scored),unscored_transitions=4833-len(scored),pixels=n,mean_rgb_difference=metrics,reduction_fraction=1-metrics['after']/max(metrics['before'],1e-6),scope='camera-compensated shared original-hand-region brown tabletop in encoded videos; excludes robot and garment, not truth accuracy',parts=[d['summary'] for d in results],joins=[d['summary'] for d in joins])
    atomic_json(a.root/'encoded_temporal_audit.json',dict(summary=summary,frames=sorted(scored,key=lambda r:r['frame_index'])))
    progress.update(4834,'completed',reduction_fraction=summary['reduction_fraction']);progress.close()
    print(json.dumps(summary))


if __name__=='__main__':main()
