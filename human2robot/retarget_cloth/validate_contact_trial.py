"""Independent pose/provenance audit; rejects trials with large tail errors."""
import argparse,json,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from human2robot.retarget_cloth.validate_preview import origin,rotation
from human2robot.object_annotation.common import Progress,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();d=json.loads((a.root/'robot_trajectory.json').read_text());rows=d['frames'];tree=ET.parse(a.root/'preview_arm.urdf').getroot();children={j.find('child').get('link'):j for j in tree.findall('joint')};chain=[];link='tcp'
    while link!='fl_base_link':j=children[link];chain.insert(0,j);link=j.find('parent').get('link')
    max_fk=0.;max_tool=0.;stats={};progress=Progress('outputs/object_annotation_20261020','robot_contact_trial_validation',len(rows)*2)
    for s in ['Left','Right']:
        errors=[];angles=[];base=np.asarray(rows[0]['robot_bases'][s]);fixed=True;qs=[]
        for i,row in enumerate(rows):
            h=row['hands'][s];q=np.asarray(h['q']);qs.append(q);T=np.eye(4);k=0
            for j in chain:
                T=T@origin(j)
                if j.get('type')!='fixed':
                    M=np.eye(4);M[:3,:3]=rotation([float(x) for x in j.find('axis').get('xyz').split()],q[k]);T=T@M;k+=1
            T=base@T;max_fk=max(max_fk,float(np.max(np.abs(T-h['actual']))));fixed&=np.allclose(base,row['robot_bases'][s]);goal=np.asarray(h['target']);flange=np.asarray(h['target_flange']);tool=np.eye(4);tool[0,3]=.145;max_tool=max(max_tool,float(np.max(np.abs(flange@tool-goal))))
            errors.append(np.linalg.norm(T[:3,3]-goal[:3,3]));angles.append(min(Rotation.from_matrix(goal[:3,:3]@T[:3,:3].T).magnitude(),Rotation.from_matrix(goal[:3,:3]@np.diag([1,-1,-1])@T[:3,:3].T).magnitude()))
            if i%240==0:progress.update((0 if s=='Left' else len(rows))+i,phase='independent_FK_and_tool_transform')
        stats[s]=dict(fixed_base=fixed,position_p95_m=float(np.percentile(errors,95)),orientation_p95_deg=float(np.degrees(np.percentile(angles,95))),orientation_median_deg=float(np.degrees(np.median(angles))),position_over_4cm_frames=int((np.array(errors)>.04).sum()),orientation_over_45deg_frames=int((np.array(angles)>np.pi/4).sum()),max_step_deg=float(np.degrees(np.abs(np.diff(qs,axis=0)).max())))
    integrity=max_fk<1e-8 and max_tool<1e-8 and all(x['fixed_base'] for x in stats.values());release=integrity and all(x['position_p95_m']<.025 and x['orientation_p95_deg']<35 for x in stats.values())
    atomic_json(a.root/'independent_validation.json',dict(integrity_pass=integrity,max_FK_delta=max_fk,max_tool_transform_delta=max_tool,metrics=stats,release_pass=release,thresholds_are_preview_criteria_not_accuracy_ground_truth=True,reason='orientation_tail_error_and_visual_arm_configuration_require_further_optimization' if not release else 'preview_gate_only',full_mesh_collision_tested=False,physics_executed=False));progress.update(len(rows)*2,'completed',release_pass=int(release));progress.close()

if __name__=='__main__':main()
