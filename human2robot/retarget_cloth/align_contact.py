"""Source-view hand grasp frames -> local cloth contacts -> flange -> arm IK.

No rigid-cloth assumption: local visible cloth is a contact proxy, not a
canonical material track. All distances inherit the uncalibrated scene scale.
"""
import argparse, json
from pathlib import Path
from collections import OrderedDict
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, distance_transform_edt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from human2robot.retarget_cloth.run_preview import Robot, transform, unit
from human2robot.retarget_cloth.stabilize_robot import continuous, smooth
from human2robot.retarget_cloth.composite_source import Inputs
from human2robot.object_annotation.common import Progress, atomic_json


def smooth_rotations(matrices, valid, sigma=5):
    inds=np.flatnonzero(valid)
    rot=Rotation.from_matrix(matrices[inds])
    allr=Slerp(inds,rot)(np.clip(np.arange(len(valid)),inds[0],inds[-1]))
    q=allr.as_quat()
    for i in range(1,len(q)):
        if q[i]@q[i-1]<0:q[i]*=-1
    q=gaussian_filter1d(q,sigma,axis=0,mode='nearest')
    return Rotation.from_quat(q/np.linalg.norm(q,axis=1,keepdims=True)).as_matrix()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    objects=Path('outputs/object_annotation_20261020');handsroot=Path('outputs/hand_mesh_repair_20261020/final')
    old=json.loads(Path('outputs/robot_repair_20261020/trajectory/robot_trajectory.json').read_text());rows=old['frames'];n=len(rows)
    source=[json.loads(l) for l in (handsroot/'frames.jsonl').read_text().splitlines()]
    assets=Path('/user/panmiao/workspace/robodojo-data-windtunnel/runtime/RoboTwin/assets/embodiments/aloha-agilex')
    robot=Robot(assets,a.root,face_budget=0);inputs=Inputs(objects,Path('outputs/robot_repair_20261020/masks_final'))
    progress=Progress(objects,'robot_contact_alignment',n*3)
    js=np.full((n,2,21,3),np.nan);uv=np.full((n,2,21,2),np.nan);cache=OrderedDict()
    try:
        for i,row in enumerate(source):
            for h in row['hands']:
                path=handsroot/h['prediction_file'];key=str(path)
                if key not in cache:
                    cache[key]=dict(np.load(path))
                    if len(cache)>8:cache.popitem(last=False)
                pack=cache[key];k=h['prediction_index'];s=0 if h['handedness']=='Left' else 1
                js[i,s]=pack['joints'][k];cam=js[i,s]+pack['cam_t'][k];uv[i,s]=cam[:,:2]/cam[:,2:]*12500+[320,240]
            if i%240==0:progress.update(i,phase='hand_grasp_and_tool_frames')
        solutions={};targets={};contact_meta={};observed={};orientations={};openings={};bases={}
        for sideidx,side in enumerate(['Left','Right']):
            valid=np.isfinite(js[:,sideidx]).all((1,2));observed[side]=valid.copy()
            rawrot=np.tile(np.eye(3),(n,1,1));gaps=np.full((n,1),np.nan)
            for i in np.flatnonzero(valid):
                j=js[i,sideidx];R=np.asarray(rows[i]['support_frame_camera']['R'])
                # Robot +X points from flange toward fingertips; +Y is jaw axis.
                approach=unit((j[4]+j[8])/2-j[0]);jaw=j[8]-j[4];jaw=unit(jaw-approach*(jaw@approach))
                if np.linalg.norm(jaw)<.9:valid[i]=False;continue
                normal=unit(np.cross(approach,jaw));rawrot[i]=R.T@np.column_stack([approach,jaw,normal]);gaps[i]=np.linalg.norm(j[8]-j[4])/2
                # Parallel jaws are symmetric about +X. Select a consistent
                # roll equivalent before interpolation, not a 180-degree flip.
                if i>0:
                    prev=rawrot[np.flatnonzero(valid[:i])[-1]] if valid[:i].any() else rawrot[i]
                    alt=rawrot[i]@np.diag([1,-1,-1])
                    if np.trace(prev.T@alt)>np.trace(prev.T@rawrot[i]):rawrot[i]=alt
            orient=smooth_rotations(rawrot,valid);orientations[side]=orient
            openings[side]=np.clip(smooth(continuous(gaps),4)[:,0],0,.04765)
            xy=smooth(continuous((uv[:,sideidx,4]+uv[:,sideidx,8])/2),3)
            oldpos=np.array([r['hands'][side]['target_position'] for r in rows]);delta=np.zeros((n,3));metas=[]
            # Sample local visible cloth. Only near-mask observations may anchor
            # a contact; missing or distant hands never invent contact labels.
            for i in range(n):
                metas.append(dict(contact_candidate=False,contact_verified=False,local_surface_available=False))
                if i%3 or not valid[i]:continue
                cloth,human=inputs.mask(i);visible=cloth&~human
                if visible.sum()<200:continue
                distance,nearest=distance_transform_edt(~visible,return_indices=True)
                x,y=np.round(xy[i]).astype(int)
                if not (0<=x<640 and 0<=y<480):continue
                near=float(distance[y,x]);cy,cx=nearest[:,y,x];depth=np.load(objects/'depth'/f'{i:06d}.npz');D=depth['depth'];Kraw=depth['K'];K=np.asarray(rows[i]['camera_K']);R=np.asarray(rows[i]['support_frame_camera']['R']);c=np.asarray(rows[i]['support_frame_camera']['origin']);normal=R[:,2];d=normal@c
                yy,xx=np.mgrid[:480,:640];patch=visible&((xx-cx)**2+(yy-cy)**2<18**2)&np.isfinite(D)
                if patch.sum()<30 or near>40:continue
                # Relative height above a robust nearby tabletop, under the
                # already stabilized plane scale. No rigid global object pose.
                bgr=inputs.rgb(i);hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);table=(hsv[:,:,0]>10)&(hsv[:,:,0]<32)&(hsv[:,:,1]>75)&~(cloth|human)&np.isfinite(D)&(yy>200)&(yy<445)
                tx,ty=xx[table][::20],yy[table][::20]
                if len(tx)<100:continue
                raytable=np.c_[tx,ty,np.ones(len(tx))]@np.linalg.inv(Kraw).T
                rawd=np.median((raytable*D[ty,tx,None])@normal)
                py,px=np.where(patch);points=np.c_[px,py,np.ones(len(px))]@np.linalg.inv(Kraw).T*D[py,px,None]*(d/rawd)
                height=float(np.clip(np.median(points@normal)-d,.004,.12))
                ray=np.linalg.inv(K)@np.r_[xy[i],1.];candidate=(ray*((d+height+.008)/(normal@ray))-c)@R
                delta[i]=np.clip(candidate-oldpos[i],-.035,.035)
                metas[i]=dict(contact_candidate=True,contact_verified=False,local_surface_available=True,nearest_visible_cloth_px=near,surface_height_estimate=height,source='visible_local_cloth_and_MANO_pinch')
            support=np.array([m['local_surface_available'] for m in metas]);idx=np.flatnonzero(support)
            if len(idx)>1:
                for j in range(3):delta[:,j]=np.interp(np.arange(n),idx,delta[idx,j])
                delta=gaussian_filter1d(delta,8,axis=0)
                # Fade local corrections away from valid measurements.
                dist=distance_transform_edt(~support);delta*=np.clip(1-dist[:,None]/30,0,1)
            target=oldpos+delta;targets[side]=target;contact_meta[side]=metas
            # One fixed initial base per arm. fit_contact_base.py searches
            # sequence-wide placements without moving the base per frame.
            base=transform(Rotation.from_euler('z',np.pi/2).as_matrix(),[-.38 if side=='Left' else .38,-.42,.015]);bases[side]=base
            inv=np.linalg.inv(base);samples=[];inds=list(range(0,n,3));inds+=[] if inds[-1]==n-1 else [n-1];previous=None
            for k,i in enumerate(inds):
                goal=transform(inv[:3,:3]@orient[i],inv[:3,:3]@target[i]+inv[:3,3])
                # Contact center is TCP; flange target is offset by the fixed
                # 145 mm URDF tool transform. Matching the flange directly to
                # the human wrist would place fingertips away from the cloth.
                seeds=[previous] if previous is not None else [np.array(rows[i]['hands'][side]['q'])]+[np.array([az,b,e,0,w,roll]) for az in [-1,1] for b,e,w in [(-1.5,1.8,-.5),(1.5,-1.8,.5)] for roll in [-1.5,1.5]]
                best=None
                for seed in seeds:
                    def residual(q):
                        T,pts=robot.robot.fk(q,True);rot=Rotation.from_matrix(goal[:3,:3]@T[:3,:3].T).as_rotvec()
                        return np.r_[(T[:3,3]-goal[:3,3])/.014,rot/.32,(q-seed)*(.25 if previous is not None else .015),np.maximum(0,.008-(pts[2:,2]+base[2,3]))/.008]
                    lo=np.full(6,-np.pi);hi=np.full(6,np.pi)
                    if previous is not None:lo=np.maximum(lo,previous-.22);hi=np.minimum(hi,previous+.22)
                    fit=least_squares(residual,np.clip(seed,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=60)
                    score=np.linalg.norm(residual(fit.x))
                    if best is None or score<best[0]:best=(score,fit.x)
                previous=best[1];samples.append(previous)
                if k%60==0:progress.update(n+int((sideidx*n+i)/1),phase=f'full_pose_IK_{side}')
            q=np.column_stack([np.interp(np.arange(n),inds,np.asarray(samples)[:,j]) for j in range(6)])
            solutions[side]=gaussian_filter1d(q,2.5,axis=0,mode='nearest')
        output=[];stats={s:dict(position=[],angle=[],old_angle=[],clearance=[]) for s in solutions}
        for i,r in enumerate(rows):
            row=dict(r);row['hands']={};row['status']='local_contact_alignment_review';row['robot_bases']={s:b.tolist() for s,b in bases.items()}
            for side in solutions:
                h=dict(r['hands'][side]);q=solutions[side][i];base=bases[side];actual=base@robot.robot.fk(q);goal=transform(orientations[side][i],targets[side][i]);_,pts=robot.robot.fk(q,True)
                angle=float(Rotation.from_matrix(goal[:3,:3]@actual[:3,:3].T).magnitude());error=float(np.linalg.norm(actual[:3,3]-goal[:3,3]));oldangle=float(Rotation.from_matrix(goal[:3,:3]@np.asarray(h['actual'])[:3,:3].T).magnitude())
                local=contact_meta[side][i];h.update(q=q.tolist(),actual=actual.tolist(),target=goal.tolist(),target_position=targets[side][i].tolist(),target_flange=(goal@transform(translation=[-.145,0,0])).tolist(),rotation_error_rad=angle,rotation_metric_available=True,error_m=error,gripper_joint_m=float(openings[side][i]),status='contact_alignment_estimate_review',link_point_table_clearance_m=float(pts[2:,2].min()+.015),contact=local)
                row['hands'][side]=h
                for key,v in [('position',error),('angle',angle),('old_angle',oldangle),('clearance',h['link_point_table_clearance_m'])]:stats[side][key].append(v)
            output.append(row)
        summary={}
        for side,d in stats.items():
            q=solutions[side];summary[side]={k:{'median':float(np.median(v)),'p95':float(np.percentile(v,95)),'max':float(np.max(v)),'min':float(np.min(v))} for k,v in d.items()};summary[side].update(max_joint_step_deg=float(np.degrees(np.abs(np.diff(q,axis=0)).max())),max_acceleration_rad_s2=float(np.abs(np.diff(q,n=2,axis=0)).max()*900))
        atomic_json(a.root/'robot_trajectory.json',dict(meta=dict(old['meta'],method='continuous MANO grasp-frame and local cloth contact proxy alignment; full-pose flange IK',robot='dual ARX5 asset with parallel grippers',tool_flange_to_tcp_m=[.145,0,0],contact_verified=False,object_global_pose_used_as_rigid_constraint=False),frames=output))
        atomic_json(a.root/'alignment_validation.json',dict(frames=n,metrics=summary,rotation_error_reference='estimated MANO grasp target, not ground truth',contact_verified=False,physics_executed=False,training_eligible=False))
        progress.update(n*3,'completed');print(json.dumps(summary))
    except Exception as e:progress.update(0,'failed',error=str(e));raise
    finally:progress.close()


if __name__=='__main__':main()
