"""EgoMatrix-inspired keypoint retargeting, without claiming physics execution.

Uses actual ARX5 STL surfaces and URDF kinematics. Observed cloth is a depth
surface, not simulated fabric. Missing detections produce no robot observation.
"""
import argparse
import copy
import json
import sys
import hashlib
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from h2r.kinematics import SerialRobot, vector
from h2r.geometry import transform
from v2.surface_renderer import render as raster, lighting
from human2robot.hand_mesh.video import frames, probe, Writer


def unit(v):
    return v / max(1e-9, np.linalg.norm(v))


def origin(e):
    if e is None:
        return np.eye(4)
    return transform(Rotation.from_euler('xyz', vector(e.get('rpy'), [0,0,0])).as_matrix(), vector(e.get('xyz'), [0,0,0]))


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    tmp.replace(path)


class Robot:
    def __init__(self, assets, output, face_budget=900):
        path = assets / 'urdf/arx5_description_isaac.urdf'
        tree = ET.parse(path).getroot()
        arm = ET.Element('robot', name='ARX5_follower_preview')
        for e in tree:
            if e.tag == 'link' and e.get('name') in ['fl_base_link'] + [f'fl_link{i}' for i in range(1,9)]:
                arm.append(copy.deepcopy(e))
            if e.tag == 'joint' and e.get('name') in [f'fl_joint{i}' for i in range(1,9)]:
                arm.append(copy.deepcopy(e))
        ET.SubElement(arm,'link',name='tcp')
        j=ET.SubElement(arm,'joint',name='tcp_fixed',type='fixed')
        ET.SubElement(j,'parent',link='fl_link6');ET.SubElement(j,'child',link='tcp')
        ET.SubElement(j,'origin',xyz='0.145 0 0')
        target=output/'preview_arm.urdf';ET.ElementTree(arm).write(target)
        self.robot=SerialRobot(target,'fl_base_link','tcp')
        # Source URDF uses simulation limits +/-10 rad and 1000 rad/s. This
        # preview uses a narrower numerical range, never hardware-safe limits.
        self.robot.lower=np.maximum(self.robot.lower,-np.pi)
        self.robot.upper=np.minimum(self.robot.upper,np.pi)
        self.joints=[j for j in arm.findall('joint') if j.get('name','').startswith('fl_joint')]
        self.meshes=[];sources=[]
        names=['base_arm']+[f'link{i}' for i in range(1,9)]
        for i,name in enumerate(names):
            p=assets/'meshes'/(name+'.STL')
            mesh=trimesh.load(p,force='mesh')
            if face_budget and len(mesh.faces)>face_budget:mesh=mesh.simplify_quadric_decimation(face_count=face_budget)
            self.meshes.append(('fl_base_link' if i==0 else f'fl_link{i}',np.asarray(mesh.vertices)[mesh.faces],np.asarray(mesh.vertex_normals)[mesh.faces]))
            sources.append(dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rendered_faces=len(mesh.faces)))
        write_json(output/'robot_asset_manifest.json',dict(urdf=str(path),urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),meshes=sources,hardware_limits_verified=False))

    def links(self,q,gap):
        result={'fl_base_link':np.eye(4)}
        for j in self.joints:
            i=int(j.get('name').replace('fl_joint',''));axis=vector(j.find('axis').get('xyz'),[1,0,0])
            motion=transform(Rotation.from_rotvec(axis*q[i-1]).as_matrix()) if i<=6 else transform(translation=axis*gap)
            result[j.find('child').get('link')]=result[j.find('parent').get('link')]@origin(j.find('origin'))@motion
        return result

    def fit(self,target,previous,dt):
        robot=self.robot
        seeds=[previous] if previous is not None else [np.array([0,-2,1.5,0,0,0]),np.array([0,2,1.5,0,0,0]),np.array([0,.6,1.3,0,-.6,0])]
        best=None
        for seed in seeds:
            def residual(q):
                actual,points=robot.fk(q,True)
                # Midpoint and two gripper-axis points mirror semantic keypoint
                # correspondences in EgoMatrix eq.6; orientation is regularized.
                local=np.array([[0,0,0],[0,.035,0],[0,-.035,0],[-.05,0,0]])
                p=local@actual[:3,:3].T+actual[:3,3]
                desired=local@target[:3,:3].T+target[:3,3]
                reg=(q-seed)*(.04 if previous is not None else .005)
                clearance=np.maximum(0,.012-(points[2:,2]+.015))/.008
                return np.r_[((p-desired)/.025).ravel(),reg,clearance]
            lo=robot.lower.copy();hi=robot.upper.copy()
            if previous is not None:
                lo=np.maximum(lo,previous-3*dt);hi=np.minimum(hi,previous+3*dt)
            q=least_squares(residual,np.clip(seed,lo+1e-7,hi-1e-7),bounds=(lo,hi),max_nfev=25,ftol=1e-4,xtol=1e-4,gtol=1e-4).x
            actual=robot.fk(q)
            err=float(np.linalg.norm(actual[:3,3]-target[:3,3]))
            rot=float(Rotation.from_matrix(target[:3,:3]@actual[:3,:3].T).magnitude())
            result=dict(q=q,actual=actual,error_m=err,rotation_error_rad=rot)
            if best is None or err+rot*.02<best['error_m']+best['rotation_error_rad']*.02:best=result
        return best


def backproject(uv,depth,K):
    x=np.clip(np.round(uv[:,0]).astype(int),0,depth.shape[1]-1)
    y=np.clip(np.round(uv[:,1]).astype(int),0,depth.shape[0]-1)
    z=[]
    for xx,yy in zip(x,y):
        patch=depth[max(0,yy-3):yy+4,max(0,xx-3):xx+4]
        valid=patch[np.isfinite(patch)&(patch>0)]
        z.append(float(np.median(valid)) if len(valid) else np.nan)
    return np.c_[uv,np.ones(len(uv))]@np.linalg.inv(K).T*np.asarray(z)[:,None]


def table_frame(bgr,depth,K):
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    mask=(hsv[:,:,0]>=12)&(hsv[:,:,0]<=29)&(hsv[:,:,1]>90)&(hsv[:,:,2]>25)
    mask[:100]=False;mask[460:]=False
    yy,xx=np.where(mask&np.isfinite(depth))
    if len(xx)<2000:return None
    uv=np.c_[xx[::20],yy[::20]];xyz=backproject(uv,depth,K)
    xyz=xyz[np.isfinite(xyz).all(1)];center=np.median(xyz,axis=0)
    for _ in range(3):
        _,_,v=np.linalg.svd(xyz-center,full_matrices=False);normal=v[-1]
        error=np.abs((xyz-center)@normal)
        xyz=xyz[error<np.percentile(error,85)]
        center=np.median(xyz,axis=0)
    if normal@center>0:normal=-normal
    right=unit(np.array([1.,0,0])-normal*normal[0]);forward=unit(np.cross(normal,right))
    R=np.column_stack([right,forward,normal])
    return R,center


def view_mesh(tri,normals,color):
    # Fixed oblique review camera: x right, y away, z up.
    eye=np.array([0.,-1.35,1.1]);look=np.array([0,.03,.12])
    forward=unit(look-eye);right=unit(np.cross(forward,[0,0,1.]));up=np.cross(right,forward)
    cam=np.column_stack([right,-up,forward])
    pts=(tri-eye)@cam
    good=(pts[:,:,2]>.03).all(1)
    pts=pts[good];normals=normals[good]
    pixels=pts[:,:,:2]/pts[:,:,2,None]*620+[320,235]
    light=unit(np.array([-.3,-.5,1.]));cols=lighting(normals,color,light)
    return pixels,pts[:,:,2],cols


def plane_mesh():
    v=np.array([[-.75,-.55,0],[.75,-.55,0],[.75,.65,0],[-.75,.65,0]])
    f=np.array([[0,1,2],[0,2,3]])
    return v[f],np.tile([0.,0,1.],(2,3,1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--objects',type=Path,required=True)
    p.add_argument('--hands',type=Path,required=True);p.add_argument('--assets',type=Path,required=True);p.add_argument('--semantics',type=Path,required=True)
    p.add_argument('--max-frames',type=int,default=0);p.add_argument('--start-frame',type=int,default=0);p.add_argument('--stride',type=int,default=3);a=p.parse_args()
    a.root.mkdir(parents=True,exist_ok=True)
    events=a.objects/'events'/('robot_preview_pilot' if a.max_frames else 'robot_preview')/a.root.name
    tb=SummaryWriter(str(events));started=time.time()
    robot=Robot(a.assets,a.root)
    handmeta=json.loads((a.hands/'manifest.json').read_text());handrows=[json.loads(l) for l in (a.hands/'frames.jsonl').read_text().splitlines()]
    objects=json.loads((a.objects/'objects.json').read_text());meta=objects['meta'];source=meta['video'];meta['frames']=len(objects['frames'])
    data=json.loads(a.semantics.read_text());transitions=[s for s in data['segments'] if s['phase']=='reset']
    total=min(a.max_frames or meta['frames'],meta['frames']-a.start_frame)
    writer=Writer(a.root/'robot_preview.mp4',640,480,30)
    previous={};last_target={};last_orientation={};last_frame={};cache={};records=[];counts={'hands':0,'matched':0,'missing_frames':0,'frames_with_robot':0,'reinitializations':0}
    bases={}
    for side,x in [('Left',-.38),('Right',.38)]:bases[side]=transform(Rotation.from_euler('z',np.pi/2).as_matrix(),[x,-.42,.015])
    tb.add_text('scope','Keypoint IK preview only. Physics/contact/PPO: waiting, not started.',0)
    tb.add_scalar('state/physics_waiting',1,0)
    tf=None
    for offset,bgr in enumerate(frames(source,meta,start=a.start_frame,limit=total)):
        i=a.start_frame+offset
        if offset%a.stride:
            row=copy.deepcopy(sample_row);row.update(frame_index=i,timestamp_s=i/30,resampled_hold=True)
            held=out.copy()
            if any(not any(h['handedness']==side for h in handrows[i]['hands']) for side in row['hands']):
                row['hands']={};row['status']='missing_hand_during_display_hold'
                held[:]=(242,242,242);cv2.putText(held,'HAND DETECTION LOST - NO ROBOT OUTPUT',(30,240),0,.55,(50,70,180),1,cv2.LINE_AA)
            records.append(row);writer.write(held);continue
        depthfile=np.load(a.objects/'depth'/f'{i:06d}.npz');depth=depthfile['depth'];K=depthfile['K']
        estimate=table_frame(bgr,depth,K)
        row={'frame_index':i,'timestamp_s':i/30,'sample_source_frame':i,'resampled_hold':False,'hands':{},'training_eligible':False,'physics_executed':False}
        transition=any(s['start_frame']<=i<s['end_frame_exclusive'] for s in transitions)
        triangles=[];normals=[];colors=[]
        def add(tri,norm,col):
            ps,z,c=view_mesh(tri,norm,col);triangles.extend(ps);normals.extend(z);colors.extend(c)
        t,n=plane_mesh();add(t,n,[175,185,190])
        if estimate is not None:
            R,center=estimate
            # The support frame is estimated independently each frame; camera
            # trajectory and real-world metric calibration remain unavailable.
            tf=estimate
            obj=objects['frames'][i];mf=a.objects/'masks'/f'{obj["chunk_start"]:06d}.npz'
            mk=str(mf)
            if mk not in cache:
                if len(cache)>12:cache.clear()
                cache[mk]=dict(np.load(mf))
            pack=cache[mk];idx=i-obj['chunk_start'];mask=np.unpackbits(pack['masks'][idx],count=640*480).reshape(480,640).astype(bool)
            # Visible cloth point surface, reprojected from each source frame;
            # gaps remain gaps and no material correspondence is asserted.
            yy,xx=np.mgrid[0:480:8,0:640:8];uv=np.c_[xx.ravel(),yy.ravel()]
            xyz=backproject(uv,depth,K);world=(xyz-center)@R
            hh,ww=yy.shape;f=[]
            valid=mask[yy,xx].ravel()&np.isfinite(world).all(1)
            for y in range(hh-1):
                for x in range(ww-1):
                    k=y*ww+x
                    for ids in [(k,k+1,k+ww),(k+1,k+ww+1,k+ww)]:
                        if valid[list(ids)].all():
                            tri=world[list(ids)]
                            if np.max(np.linalg.norm(tri-np.roll(tri,1,axis=0),axis=1))<.08:f.append(ids)
            if f:
                tri=world[np.asarray(f)];n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);n/=np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-9);n[n[:,2]<0]*=-1
                add(tri,np.repeat(n[:,None,:],3,axis=1),[220,223,224])
            for h in handrows[i]['hands']:
                side=h['handedness'];key=str(a.hands/h['prediction_file'])
                if key not in cache:
                    if len(cache)>12:cache.clear()
                    cache[key]=dict(np.load(key))
                saved=cache[key];j=h['prediction_index'];local=saved['joints'][j];xyz=local+saved['cam_t'][j]
                uv=xyz[:,:2]/xyz[:,2:]*handmeta['focal_length_px']+[320,240]
                if not ((uv[[0,4,8,9]]>=0)&(uv[[0,4,8,9]]<[640,480])).all():continue
                pts=(backproject(uv,depth,K)-center)@R
                if not np.isfinite(pts).all():continue
                tcp=(pts[4]+pts[8])/2
                # Hand-local articulation supplies orientation; scene-depth
                # discontinuities at fingertips must not define wrist rotation.
                handlocal=local@R
                approach=unit(handlocal[[5,9,13,17]].mean(0)-handlocal[0]);jaw=handlocal[8]-handlocal[4];jaw=unit(jaw-approach*(jaw@approach))
                z=unit(np.cross(approach,jaw));jaw=unit(np.cross(z,approach))
                if np.linalg.norm(z)<.9:continue
                orient=np.column_stack([approach,jaw,z])
                reset=bool(transition or side not in last_frame or i-last_frame[side]>a.stride or (side in last_target and np.linalg.norm(tcp-last_target[side])>.15))
                raw_tcp=tcp.copy()
                if not reset:
                    tcp=.45*tcp+.55*last_target[side]
                    rotstep=Rotation.from_matrix(orient@last_orientation[side].T).as_rotvec()
                    orient=Rotation.from_rotvec(.35*rotstep).as_matrix()@last_orientation[side]
                T=transform(orient,tcp)
                prev=None if reset else previous.get(side)
                solution=robot.fit(np.linalg.inv(bases[side])@T,prev,a.stride/30)
                q=solution['q'];previous[side]=q;last_target[side]=tcp;last_orientation[side]=orient;last_frame[side]=i
                gap=float(np.linalg.norm(pts[8]-pts[4]));opening=float(np.clip(gap/2,0,.04765))
                actual=bases[side]@solution['actual']
                match=solution['error_m']<.035 and solution['rotation_error_rad']<.65
                counts['hands']+=1;counts['matched']+=int(match);counts['reinitializations']+=int(reset)
                _,lp=robot.robot.fk(q,True);clearance=float(np.min(lp[2:,2]+.015))
                row['hands'][side]=dict(q=q.tolist(),estimated_gap_m=gap,gripper_joint_m=opening,target=T.tolist(),actual=actual.tolist(),raw_target_position=raw_tcp.tolist(),target_smoothing_displacement_m=float(np.linalg.norm(tcp-raw_tcp)),error_m=solution['error_m'],rotation_error_rad=solution['rotation_error_rad'],link_point_table_clearance_m=clearance,status='kinematic_match_review' if match else 'tracking_error_review',reinitialized=reset,source_hand_quality=h['quality']['status'])
                links=robot.links(q,opening)
                for link,tri,n in robot.meshes:
                    worldT=bases[side]@links[link];worldtri=tri@worldT[:3,:3].T+worldT[:3,3];wn=n@worldT[:3,:3].T
                    color=[160,125,72] if side=='Left' else [145,98,163]
                    if link.endswith(('7','8')):color=[195,202,211]
                    add(worldtri,wn,color)
            row['support_frame_camera']=dict(R=R.tolist(),origin=center.tolist())
        row['status']='preview' if row['hands'] else 'missing_hand_or_support'
        counts['frames_with_robot']+=int(bool(row['hands']));counts['missing_frames']+=int(not row['hands'])
        image,mask=raster(np.asarray(triangles),np.asarray(normals),np.asarray(colors),640,480,True)
        bg=np.full((480,640,3),242,np.uint8);alpha=mask[:,:,None]/255.;out=(image*alpha+bg*(1-alpha)).astype(np.uint8)
        cv2.rectangle(out,(0,0),(640,58),(36,42,48),-1)
        cv2.putText(out,'Dual ARX5 | keypoint retargeting trial',(10,22),0,.6,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(out,f'{i/30:.1f}s | {30/a.stride:g}Hz motion / 30fps | NO PHYSICS / NOT ACTION LABELS',(10,45),0,.42,(100,210,255),1,cv2.LINE_AA)
        if not row['hands']:cv2.putText(out,'HAND OR SUPPORT DATA MISSING',(100,250),0,.6,(50,70,180),2,cv2.LINE_AA)
        errors=' '.join(f'{s[0]}: {r["error_m"]*100:.1f}cm' for s,r in row['hands'].items())
        cv2.putText(out,'Tracking residual '+errors,(10,470),0,.45,(55,65,70),1,cv2.LINE_AA)
        if any(h['status']=='tracking_error_review' for h in row['hands'].values()):cv2.putText(out,'TRACKING ERROR - REVIEW',(10,86),0,.48,(40,80,180),1,cv2.LINE_AA)
        if any(h['link_point_table_clearance_m']<0 for h in row['hands'].values()):cv2.putText(out,'LINK POINT BELOW TABLE - REVIEW',(10,106),0,.48,(40,80,180),1,cv2.LINE_AA)
        writer.write(out);records.append(row);sample_row=row
        if offset%120==0 or offset==total-1:
            status=dict(done=offset+1,total=total,status='running',elapsed_s=time.time()-started,**counts)
            write_json(a.root/'status.json',status)
            tb.add_scalar('progress/frames',offset+1,offset+1);tb.add_scalar('progress/completion',(offset+1)/total,offset+1)
            tb.add_scalar('quality/matched_keypoint_hands',counts['matched'],offset+1);tb.add_scalar('quality/missing_frames',counts['missing_frames'],offset+1);tb.flush()
            print(json.dumps(status),flush=True)
            cv2.imwrite(str(a.root/f'preview_{i:06d}.jpg'),out)
    writer.close()
    errors=[h['error_m'] for r in records for h in r['hands'].values()]
    write_json(a.root/'robot_trajectory.json',dict(meta=dict(fps=30,motion_sample_fps=30/a.stride,render_resampling='zero-order hold, missing detections remain unavailable',frames=total,start_frame=a.start_frame,source_sha256=handmeta['source_sha256'],robot='dual Aloha / AgileX ARX5',method='EgoMatrix-inspired semantic keypoint IK, not full reproduction',training_eligible=False,physics_executed=False,camera_motion_calibrated=False,metric_scale_verified=False,support_frame='per-frame MoGe table-plane estimate',limits='numerical +/-pi, 3rad/s within contiguous detections; not hardware limits',reference='https://arxiv.org/html/2610.18822v1#S3.SS2'),frames=records))
    quality=dict(**counts,count_units='solved motion samples (held display frames excluded)',frames=total,display_frames_with_robot=sum(bool(r['hands']) for r in records),motion_sample_fps=30/a.stride,position_error_median=float(np.median(errors)) if errors else None,position_error_p95=float(np.percentile(errors,95)) if errors else None,physics_executed=False,collision_verified=False,cloth_simulated=False,training_eligible=False,model_accuracy_verified=False)
    write_json(a.root/'quality.json',quality);write_json(a.root/'status.json',dict(done=total,total=total,status='completed',**counts))
    tb.add_scalar('progress/completion',1,total);tb.add_text('status','Completed kinematic preview; physics and contact remain unverified.',total);tb.close();print(json.dumps(quality),flush=True)


if __name__=='__main__':main()
