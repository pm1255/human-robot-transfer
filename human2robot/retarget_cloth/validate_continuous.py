"""Independent FK and display continuity audit, with missing evidence checks."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from human2robot.retarget_cloth.validate_preview import origin,rotation
from human2robot.object_annotation.common import Progress,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();data=json.loads((a.root/'robot_trajectory.json').read_text());rows=data['frames'];source=[json.loads(l) for l in Path('outputs/hand_mesh_repair_20261020/final/frames.jsonl').read_text().splitlines()]
    tree=ET.parse(a.root/'preview_arm.urdf').getroot();children={j.find('child').get('link'):j for j in tree.findall('joint')};chain=[];link='tcp'
    while link!='fl_base_link':j=children[link];chain.insert(0,j);link=j.find('parent').get('link')
    monitor=Progress('outputs/object_annotation_20261020','robot_repair_independent_validation',len(rows));failures=[];max_delta=0.;p2d={s:[] for s in ['Left','Right']};clear=[]
    for i,row in enumerate(rows):
        if set(row['hands'])!={'Left','Right'}:failures.append([i,'missing_arm'])
        for side,h in row['hands'].items():
            q=np.asarray(h['q']);T=np.eye(4);active=0;points=[]
            for j in chain:
                T=T@origin(j)
                if j.get('type')!='fixed':
                    M=np.eye(4);M[:3,:3]=rotation([float(x) for x in j.find('axis').get('xyz').split()],q[active]);T=T@M;active+=1
                points.append(T[:3,3])
            base=np.eye(4);base[:3,:3]=rotation([0,0,1],np.pi/2);base[:3,3]=[-.38 if side=='Left' else .38,-.42,.015];T=base@T;delta=np.max(np.abs(T-h['actual']));max_delta=max(max_delta,float(delta));clear.append(float(np.min(np.asarray(points)[1:,2]+.015)))
            observed=any(x['handedness']==side for x in source[i]['hands'])
            if delta>1e-8 or np.abs(q).max()>np.pi+1e-7 or h['source_hand_observed']!=observed or h['display_inferred']==observed:failures.append([i,side,'FK_or_provenance'])
            R=np.asarray(row['support_frame_camera']['R']);c=np.asarray(row['support_frame_camera']['origin']);K=np.asarray(row['camera_K']);cam=R@T[:3,3]+c;pixel=K@cam;p2d[side].append(pixel[:2]/pixel[2])
        if i%240==0:monitor.update(i,phase='independent_fk_and_continuity')
    metrics={}
    for side in p2d:
        q=np.array([r['hands'][side]['q'] for r in rows]);step=np.abs(np.diff(q,axis=0));v=step*30;acc=np.abs(np.diff(q,n=2,axis=0))*900;px=np.linalg.norm(np.diff(p2d[side],axis=0),axis=1)
        metrics[side]=dict(max_frame_step_deg=float(np.degrees(step.max())),max_speed_rad_s=float(v.max()),max_acceleration_rad_s2=float(acc.max()),tcp_pixel_step_p95=float(np.percentile(px,95)),tcp_pixel_step_max=float(px.max()))
        if step.max()>.061 or v.max()>1.81:failures.append([side,'display_speed_limit'])
    atomic_json(a.root/'independent_validation.json',dict(integrity_pass=not failures,failures=failures,independent_fk_max_delta=max_delta,continuity=metrics,minimum_link_point_clearance_m=min(clear),link_point_below_table_samples=sum(c<0 for c in clear),source_frame_count=len(source),frames=len(rows),complete_two_arm_records=all(len(r['hands'])==2 for r in rows),physics_executed=False,training_eligible=False))
    monitor.update(len(rows),'failed' if failures else 'completed');monitor.close();print(json.dumps(metrics))
    if failures:raise SystemExit(1)

if __name__=='__main__':main()
