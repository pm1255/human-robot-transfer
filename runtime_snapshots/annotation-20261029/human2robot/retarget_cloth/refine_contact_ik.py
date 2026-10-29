"""Refine contact-pose IK with unwrapped angular continuity and table clearance."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from human2robot.retarget_cloth.run_preview import Robot,transform
from human2robot.object_annotation.common import Progress,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--tool-clearance',action='store_true');a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    data=json.loads(a.input.read_text());rows=data['frames'];n=len(rows);robot=Robot(Path('/user/panmiao/workspace/robodojo-data-windtunnel/runtime/RoboTwin/assets/embodiments/aloha-agilex'),a.root,face_budget=0);progress=Progress('outputs/object_annotation_20261020','robot_contact_ik_refine',n*2)
    stats={};recorded={}
    try:
        for si,side in enumerate(['Left','Right']):
            if a.tool_clearance:
                for row in rows:
                    h=row['hands'][side];goal=np.asarray(h['target']);R=np.asarray(row['support_frame_camera']['R']);c=np.asarray(row['support_frame_camera']['origin']);camera=R@goal[:3,3]+c;ray=camera/camera[2];normal=R[:,2];d=normal@c
                    clearance=max(.023+.145*goal[2,0],.012+abs(goal[2,1])*h['gripper_joint_m'])
                    height=max(goal[2,3],clearance)
                    pos=(ray*((d+height)/(normal@ray))-c)@R
                    h['pre_tool_clearance_target']=goal.tolist();goal[:3,3]=pos;h['target']=goal.tolist();h['target_position']=pos.tolist();h['target_flange']=(goal@transform(translation=[-.145,0,0])).tolist()
            base=np.asarray(rows[0]['robot_bases'][side]);inv=np.linalg.inv(base);inds=list(range(0,n,3))+[n-1];qs={};seed=np.asarray(rows[120]['hands'][side]['q']);pivot=inds.index(120)
            for sequence in [inds[pivot:],list(reversed(inds[:pivot]))]:
                previous=seed if sequence[0]==120 else qs[120]
                for i in sequence:
                    goal=inv@np.asarray(rows[i]['hands'][side]['target'])
                    def residual(q):
                        T,pts=robot.robot.fk(q,True)
                        rot1=Rotation.from_matrix(goal[:3,:3]@T[:3,:3].T).as_rotvec();rot2=Rotation.from_matrix(goal[:3,:3]@np.diag([1,-1,-1])@T[:3,:3].T).as_rotvec();rot=rot1 if np.linalg.norm(rot1)<=np.linalg.norm(rot2) else rot2
                        return np.r_[(T[:3,3]-goal[:3,3])/.012,rot/.20,(q-previous)*.12,np.maximum(0,.018-(pts[2:,2]+base[2,3]))/.002]
                    # +/- pi was an arbitrary preview range, not a mechanical
                    # stop. Use an unwrapped numerical representation to avoid
                    # forcing a full revolution when crossing that boundary.
                    lo=np.maximum(-3*np.pi,previous-.18);hi=np.minimum(3*np.pi,previous+.18)
                    fit=least_squares(residual,np.clip(previous,lo+1e-8,hi-1e-8),bounds=(lo,hi),max_nfev=55)
                    qs[i]=fit.x;previous=fit.x
                    if len(qs)%80==0:progress.update(si*n+min(n,len(qs)*3),phase=f'continuous_flange_IK_{side}')
            q=np.column_stack([np.interp(np.arange(n),inds,np.array([qs[i] for i in inds])[:,j]) for j in range(6)])
            q=gaussian_filter1d(q,4,axis=0,mode='nearest');errs=[];angs=[];clear=[];oldangs=[]
            for i,row in enumerate(rows):
                h=row['hands'][side];goal=np.asarray(h['target']);T,pts=robot.robot.fk(q[i],True);actual=base@T;err=float(np.linalg.norm(goal[:3,3]-actual[:3,3]));ang=min(float(Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T).magnitude()),float(Rotation.from_matrix(goal[:3,:3]@np.diag([1,-1,-1])@actual[:3,:3].T).magnitude()));c=float(pts[2:,2].min()+base[2,3]);errs.append(err);angs.append(ang);clear.append(c)
                h.update(q=q[i].tolist(),actual=actual.tolist(),error_m=err,rotation_error_rad=ang,link_point_table_clearance_m=c)
            recorded[side]=q;stats[side]=dict(position_median_m=float(np.median(errs)),position_p95_m=float(np.percentile(errs,95)),position_max_m=max(errs),orientation_median_deg=float(np.degrees(np.median(angs))),orientation_p95_deg=float(np.degrees(np.percentile(angs,95))),minimum_link_point_clearance_m=min(clear),max_joint_step_deg=float(np.degrees(np.abs(np.diff(q,axis=0)).max())),max_joint_acceleration_rad_s2=float(np.abs(np.diff(q,n=2,axis=0)).max()*900),frames_position_over_4cm=int((np.array(errs)>.04).sum()),frames_rotation_over_45deg=int((np.array(angs)>np.pi/4).sum()))
        data['meta'].update(numerical_angle_range_rad=[-3*np.pi,3*np.pi],hardware_limits_verified=False,orientation_metric='parallel jaw roll symmetry aware geodesic distance to estimated target',tool_clearance_adjusted=a.tool_clearance or data['meta'].get('tool_clearance_adjusted',False),joint_representation='unwrapped continuous angles; hardware configuration requires separate validation')
        atomic_json(a.root/'robot_trajectory.json',data);atomic_json(a.root/'alignment_validation.json',dict(frames=n,metrics=stats,contact_verified=False,physics_executed=False,training_eligible=False));progress.update(n*2,'completed');print(json.dumps(stats))
    except Exception as e:progress.update(0,'failed',error=str(e));raise
    finally:progress.close()

if __name__=='__main__':main()
