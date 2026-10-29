"""Continuous display trajectory; inferred gaps remain explicitly annotated.

This is not a hardware controller or a physics demonstration.
"""
import argparse,json
from pathlib import Path
import cv2,numpy as np
from scipy.ndimage import median_filter,gaussian_filter1d
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from human2robot.retarget_cloth.run_preview import Robot,transform,unit
from human2robot.object_annotation.common import Progress,atomic_json

def smooth(x,sigma=7):
    return gaussian_filter1d(median_filter(x,size=(9,1),mode='nearest'),sigma,axis=0,mode='nearest')

def continuous(x):
    y=x.copy();idx=np.arange(len(x))
    for j in range(x.shape[1]):
        valid=np.isfinite(x[:,j]);y[:,j]=np.interp(idx,idx[valid],x[valid,j])
    return y

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--pixels',type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    objects=Path('outputs/object_annotation_20261020');old=json.load(open('outputs/semantic_robot_20261020/full_v3/robot_trajectory.json'));n=len(old['frames']);uv=np.load(a.pixels)['joints_uv']
    assets=Path('/user/panmiao/workspace/robodojo-data-windtunnel/runtime/RoboTwin/assets/embodiments/aloha-agilex');robot=Robot(assets,a.root,face_budget=0);progress=Progress(objects,'robot_continuous_trajectory',n)
    # Plane center computed from "visible brown pixels" wandered laterally by
    # >1 m when the garment moved. Use a fixed image anchor and smooth plane
    # normal; fix the uncalibrated scale and focal length for this display.
    normals=np.full((n,3),np.nan);plane=[];focals=[]
    for i,row in enumerate(old['frames']):
        if 'support_frame_camera' in row:
            f=row['support_frame_camera'];normals[i]=np.asarray(f['R'])[:,2];plane.append(float(normals[i]@f['origin']))
        if i%30==0:focals.append(float(np.load(objects/'depth'/f'{i:06d}.npz')['K'][0,0]))
    normals=smooth(continuous(normals),12);normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    K=np.array([[np.median(focals),0,319.5],[0,np.median(focals),239.5],[0,0,1.]])
    # Retain the camera pitch signal while removing uncertain metric-scale
    # flicker. Ray anchor is the central table working area, not cloth centroid.
    d=float(np.median(plane));anchor=np.linalg.inv(K)@np.array([319.5,310.,1.]);cameras=[]
    for normal in normals:
        right=unit(np.array([1.,0,0])-normal*normal[0]);forward=unit(np.cross(normal,right));R=np.column_stack([right,forward,normal]);center=anchor*(d/(normal@anchor));cameras.append((R,center))
    targets={};observed={};gaps={};solutions={};openings={}
    try:
        for sideidx,side in enumerate(['Left','Right']):
            observed[side]=np.isfinite(uv[:,sideidx]).all(axis=(1,2));pinch=(uv[:,sideidx,4]+uv[:,sideidx,8])/2
            pinch=smooth(continuous(pinch),4)
            height=np.full((n,1),np.nan);opening=np.full((n,1),np.nan)
            for i,row in enumerate(old['frames']):
                h=row['hands'].get(side)
                if h:height[i]=np.clip(h['target'][2][3],.015,.10);opening[i]=h['gripper_joint_m']
            height=smooth(continuous(height),10)[:,0];opening=smooth(continuous(opening),4)[:,0];openings[side]=opening
            xyz=[]
            for i,(R,center) in enumerate(cameras):
                ray=np.linalg.inv(K)@np.r_[pinch[i],1.];point=ray*((d+height[i])/(normals[i]@ray));xyz.append((point-center)@R)
            xyz=smooth(np.asarray(xyz),5);xyz[:,0]=np.clip(xyz[:,0],-.45,.45);xyz[:,1]=np.clip(xyz[:,1],-.32,.5);targets[side]=xyz
            base=transform(Rotation.from_euler('z',np.pi/2).as_matrix(),[-.38 if side=='Left' else .38,-.42,.015]);inv=np.linalg.inv(base)
            previous=None;sample=[];inds=list(range(0,n,3));inds.append(n-1) if inds[-1]!=n-1 else None
            for k,i in enumerate(inds):
                target=inv[:3,:3]@xyz[i]+inv[:3,3];axis=inv[:3,:3]@np.array([0.,0.,-1.])
                seeds=[previous] if previous is not None else [np.array([0,-1.6,1.8,0,1,0.]),np.array([0,1.6,-1.8,0,-1,0.])]
                best=None
                for seed in seeds:
                    def residual(q):
                        T,pts=robot.robot.fk(q,True)
                        return np.r_[(T[:3,3]-target)/.025,(T[:3,0]-axis)*1.2,(q-seed)*.35,np.maximum(0,.012-(pts[2:,2]+.015))/.012]
                    lo=np.full(6,-np.pi);hi=np.full(6,np.pi)
                    if previous is not None:lo=np.maximum(lo,previous-.18);hi=np.minimum(hi,previous+.18)
                    fit=least_squares(residual,np.clip(seed,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=35)
                    score=np.linalg.norm(residual(fit.x))
                    if best is None or score<best[0]:best=(score,fit.x)
                previous=best[1];sample.append(previous)
                if k%60==0:progress.update(int((sideidx*n+i)/2),phase=f'continuous_IK_{side}')
            q=np.column_stack([np.interp(np.arange(n),inds,np.asarray(sample)[:,j]) for j in range(6)])
            q=gaussian_filter1d(q,4,axis=0,mode='nearest')
            solutions[side]=q
        rows=[];errors=[]
        for i in range(n):
            R,center=cameras[i];row=dict(frame_index=i,timestamp_s=i/30,hands={},status='continuous_display_estimate',camera_K=K.tolist(),support_frame_camera=dict(R=R.tolist(),origin=center.tolist()),physics_executed=False,training_eligible=False)
            for side in ['Left','Right']:
                q=solutions[side][i];base=transform(Rotation.from_euler('z',np.pi/2).as_matrix(),[-.38 if side=='Left' else .38,-.42,.015]);actual=base@robot.robot.fk(q);err=float(np.linalg.norm(actual[:3,3]-targets[side][i]));_,lp=robot.robot.fk(q,True)
                row['hands'][side]=dict(q=q.tolist(),gripper_joint_m=float(openings[side][i]),actual=actual.tolist(),target_position=targets[side][i].tolist(),error_m=err,rotation_error_rad=None,rotation_metric_available=False,link_point_table_clearance_m=float(lp[2:,2].min()+.015),reinitialized=False,status='continuous_estimate_review',source_hand_observed=bool(observed[side][i]),display_inferred=not bool(observed[side][i]))
                errors.append(err)
            rows.append(row)
        metrics={}
        for side,q in solutions.items():
            dq=np.diff(q,axis=0)*30;ddq=np.diff(dq,axis=0)*30;metrics[side]=dict(max_step_rad=float(np.abs(np.diff(q,axis=0)).max()),max_speed_rad_s=float(np.abs(dq).max()),max_acceleration_rad_s2=float(np.abs(ddq).max()),inferred_frames=int((~observed[side]).sum()))
        atomic_json(a.root/'robot_trajectory.json',dict(meta=dict(frames=n,fps=30,method='continuous constrained IK on stabilized scene; missing detections interpolated for display only',training_eligible=False,physics_executed=False,camera_motion_calibrated=False),frames=rows))
        atomic_json(a.root/'trajectory_validation.json',dict(frames=n,arms_per_frame=2,joints=metrics,error_median=float(np.median(errors)),error_p95=float(np.percentile(errors,95)),focal_px=float(K[0,0]),plane_distance_m=d,scale_calibrated=False,physics_executed=False,training_eligible=False))
        progress.update(n,'completed',phase='continuous_trajectory_saved')
    except Exception as e:progress.update(0,'failed',error=str(e));raise
    finally:progress.close()

if __name__=='__main__':main()
