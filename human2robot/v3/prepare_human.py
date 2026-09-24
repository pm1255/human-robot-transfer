"""Only EPIC RGB is read; all poses are inferred here. No hardware human labels."""
import argparse, json, subprocess, multiprocessing as mpool, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from v3.common import *


def candidates():
    ds=pd.concat([pd.read_parquet(p) for p in sorted((HUMAN/'meta/episodes').rglob('*.parquet'))])
    out=[]
    fps=json.loads((HUMAN/'meta/info.json').read_text())['fps']
    for _,r in ds.iterrows():
        if bool(r.get('is_third_person',False)) or bool(r.get('non_operator_hands',False)): continue
        for j,a in enumerate(r.action_config):
            if not category(a['action_text']):continue
            start=int(a['start_frame'])/fps; end=int(a['end_frame'])/fps
            if end-start<1.2:continue
            out.append(dict(id=f"epic_{int(r.episode_index):06d}_{j:02d}",episode=int(r.episode_index),
                source=str(HUMAN/r.video_path),start=start+float(r['videos/observation.images.head/from_timestamp']),
                duration=min(end-start,30),text=a['action_text'],category='container_transfer',
                split=split_episode('epic',r.video_path),annotation_source='upstream_vlm_unverified',
                hardware_labels_consumed=False))
    # Expand with RGB-only Ego4D; identical atomic container-transfer criteria.
    ego=Path('/user/zhangxueqian/dataset/lerobot_v3.0_v0/ego/ego4d/ego4d_clip_20s_lda_clean_nongrade45_new')
    info=json.loads((ego/'meta/info.json').read_text())
    cols=['episode_index','action_config','is_third_person','non_operator_hands','stage1_label',
          'videos/observation.images.head/chunk_index','videos/observation.images.head/file_index',
          'videos/observation.images.head/from_timestamp']
    for path in sorted((ego/'meta/episodes').rglob('*.parquet')):
        df=pd.read_parquet(path,columns=cols)
        for _,r in df.iterrows():
            if r.stage1_label not in ['action','navigation'] or r.is_third_person or r.non_operator_hands:continue
            source=ego/info['video_path'].format(video_key='observation.images.head',chunk_index=int(r[cols[-3]]),file_index=int(r[cols[-2]]))
            for j,a in enumerate(r.action_config):
                if not category(a['action_text']):continue
                start=int(a['start_frame'])/info['fps'];end=int(a['end_frame'])/info['fps']
                if end-start<1.2:continue
                ep=int(r.episode_index)
                out.append(dict(id=f'ego4d_{ep:06d}_{j:02d}',episode=1000000+ep,source=str(source),
                    start=start+float(r[cols[-1]]),duration=min(end-start,30),text=a['action_text'],
                    category='container_transfer',split=split_episode('ego4d',str(source)),
                    annotation_source='upstream_vlm_unverified',hardware_labels_consumed=False))
    return sorted(out,key=lambda x:hashlib.sha256(x['id'].encode()).hexdigest())



def infer_pose(xy, world, shape):
    import cv2
    h,w=shape[:2]; pix=xy*np.array([w,h]); k=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],np.float64)
    # Assumed focal length, NEVER presented as calibrated metric ground truth.
    ids=np.array([0,1,2,5,9,13,17]); obj=world[ids].astype(np.float64)
    # Translation-only fit prevents planar-palm PnP flips; camera axes come from MediaPipe.
    uv=(pix[ids]-[w/2,h/2])/w
    A=np.zeros((len(ids)*2,3)); rhs=np.zeros(len(ids)*2)
    A[::2,0]=1;A[1::2,1]=1;A[::2,2]=-uv[:,0];A[1::2,2]=-uv[:,1]
    rhs[::2]=uv[:,0]*obj[:,2]-obj[:,0];rhs[1::2]=uv[:,1]*obj[:,2]-obj[:,1]
    tv=np.linalg.lstsq(A,rhs,rcond=None)[0]
    pts=world+tv
    proj=pts[ids,:2]/pts[ids,2:3]*w+[w/2,h/2]
    err=float(np.linalg.norm(proj-pix[ids],axis=1).mean()/w)
    if err>.035 or not (.12<pts[:,2].mean()<2.5):return None
    # Stable palm axes, independent of thumb/index tips approaching at closure.
    x=pts[5]-pts[17]
    if not np.isfinite(pts).all() or np.linalg.norm(x)<1e-6:return None
    x/=np.linalg.norm(x)
    z=pts[9]-pts[0]; z-=x*np.dot(z,x)
    if np.linalg.norm(z)<.005:return None
    z/=np.linalg.norm(z); y=np.cross(z,x); rot=np.stack([x,y,z],1)
    width=np.linalg.norm(pts[5]-pts[17]); opening=np.clip(np.linalg.norm(pts[4]-pts[8])/(width*1.5),0,1)
    pose=np.r_[(pts[4]+pts[8])/2,Rotation.from_matrix(rot).as_quat(),opening].astype(np.float32)
    return pose,err


def process(row):
    import cv2, mediapipe as mp
    cv2.setNumThreads(1)
    out=ROOT/'human_stable'; out.mkdir(parents=True,exist_ok=True); dest=out/(row['id']+'.npz'); meta=dest.with_suffix('.json')
    if dest.exists() and meta.exists():
        saved=json.loads(meta.read_text())
        if saved.get('algorithm_version')==5:return saved
    try:
        raw=subprocess.check_output(['ffmpeg','-v','error','-filter_threads','1','-filter_complex_threads','1','-threads','1','-ss',str(row['start']),'-i',row['source'],'-t',str(row['duration']),'-vf','scale=512:288:force_original_aspect_ratio=decrease,pad=512:288:(ow-iw)/2:(oh-ih)/2,fps=5','-an','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],timeout=120)
        frames=np.frombuffer(raw,np.uint8).reshape(-1,288,512,3)
        V=mp.tasks.vision
        opt=V.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(ROOT/'models/hand_landmarker.task')),running_mode=V.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.4,min_hand_presence_confidence=.4,min_tracking_confidence=.4)
        images=[]; poses=[]; valid=[]; errors=[]; flow=[]; previous=None
        with V.HandLandmarker.create_from_options(opt) as model:
            for i,rgb in enumerate(frames):
                rgb=np.ascontiguousarray(rgb); result=model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb),i*200)
                pose=np.zeros((2,8),np.float32); pose[:,6]=1; mask=np.zeros(2,bool); err=np.full(2,1.)
                counts=[sum(h[0].category_name==side for h in result.handedness) for side in ['Left','Right']]
                handmask=np.ones((288,512),np.uint8)*255
                for j,h in enumerate(result.handedness):
                    side=int(h[0].category_name=='Right')
                    if counts[side]!=1:continue
                    xy=np.array([[p.x,p.y] for p in result.hand_landmarks[j]])
                    world=np.array([[p.x,p.y,p.z] for p in result.hand_world_landmarks[j]])
                    lo=np.maximum((xy.min(0)*[512,288]-30).astype(int),0); hi=np.minimum((xy.max(0)*[512,288]+30).astype(int),[512,288]);handmask[lo[1]:hi[1],lo[0]:hi[0]]=0
                    critical=xy[[0,4,5,8,9,17]]
                    if np.any(critical<-.06) or np.any(critical>1.06):continue
                    p=infer_pose(xy,world,rgb.shape)
                    if p is not None:pose[side],err[side]=p;mask[side]=True
                gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY); movement=0.
                if previous is not None:
                    features=cv2.goodFeaturesToTrack(previous,100,.02,8,mask=handmask)
                    movement=1.
                    if features is not None and len(features)>=10:
                        q,status,_=cv2.calcOpticalFlowPyrLK(previous,gray,features,None)
                        ok=status[:,0].astype(bool)
                        if ok.sum()>=10:movement=float(np.median(np.linalg.norm(q[ok,0]-features[ok,0],axis=1))/512)
                previous=gray;flow.append(movement)
                images.append(cv2.resize(rgb,(SIZE,SIZE),interpolation=cv2.INTER_AREA));poses.append(pose);valid.append(mask);errors.append(err)
        poses=np.array(poses);valid=np.array(valid);flow=np.array(flow); windows=[]
        observed=valid.copy();imputed=np.zeros_like(valid)
        from scipy.spatial.transform import Slerp
        for side in range(2):
            for i in range(1,len(poses)-1):
                if observed[i,side] or not observed[i-1,side] or not observed[i+1,side]:continue
                a,b=poses[i-1,side],poses[i+1,side]
                angle=(Rotation.from_quat(a[3:7]).inv()*Rotation.from_quat(b[3:7])).magnitude()
                if np.linalg.norm(a[:3]-b[:3])>.08 or angle>.7 or abs(a[7]-b[7])>.3:continue
                poses[i,side,:3]=(a[:3]+b[:3])/2
                poses[i,side,3:7]=Slerp([0,1],Rotation.from_quat([a[3:7],b[3:7]]))([.5]).as_quat()[0]
                poses[i,side,7]=(a[7]+b[7])/2;valid[i,side]=True;imputed[i,side]=True
        window_masks=[]
        for t in range(len(frames)-1):
            for side in range(2):
                if not observed[t,side]:continue
                mask=np.zeros(HORIZON,bool)
                for k in [1]:
                    if not observed[t+k,side]:continue
                    # Missing targets have zero loss weight; never manufacture a complete sequence.
                    if not valid[t:t+k+1,side].all() or imputed[t:t+k+1,side].sum()>1:continue
                    pp=poses[t:t+k+1,side]
                    dp=np.linalg.norm(np.diff(pp[:,:3],axis=0),axis=1)
                    dr=(Rotation.from_quat(pp[:-1,3:7]).inv()*Rotation.from_quat(pp[1:,3:7])).magnitude()
                    if (dp>.15).any() or (dr>1.2).any() or (flow[t+1:t+k+1]>.045).any():continue
                    mask[k-1]=True
                if mask.any():windows.append([t,side]);window_masks.append(mask)
        np.savez_compressed(dest,images=np.array(images),poses=poses,valid=valid,observed=observed,imputed=imputed,reprojection=np.array(errors),background_flow=flow,windows=np.array(windows,dtype=np.int32).reshape(-1,2),window_masks=np.array(window_masks,dtype=bool).reshape(-1,HORIZON))
        rec={**row,'algorithm_version':5,'file':dest.name,'frames':len(frames),'windows':len(windows),'sha256':sha(dest),'pose_provenance':'MediaPipe RGB camera-oriented geometry + assumed-intrinsics translation fit; weak pseudo TCP; not calibrated or executable ground truth','gripper_provenance':'thumb-index gap divided by 1.5*palm width, clipped; not contact or force','robot_training_valid':False,'weak_pretrain_eligible':True}
        dump(meta,rec);return rec
    except Exception as e:
        return {**row,'error':repr(e),'trace':traceback.format_exc()[-1200:],'windows':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    rows=candidates();dump(ROOT/'human_candidates.json',rows);print('CANDIDATES',len(rows),flush=True)
    records=[]
    with mpool.get_context('spawn').Pool(a.workers,maxtasksperchild=8) as pool:
        for r in pool.imap(process,rows):
            records.append(r); print(r['id'],r['windows'],r.get('error',''),flush=True)
            if len(records)%20==0:dump(ROOT/'human_stable_progress.json',{'processed':len(records),'total':len(rows),'windows':sum(x['windows'] for x in records)})
            if sum(min(x['windows'],100) for x in records if x['split']=='train')>=12000 and sum(x['windows'] for x in records if x['split']=='validation')>=800 and sum(x['windows'] for x in records if x['split']=='test')>=800:break
    dump(ROOT/'human_stable_manifest.json',sorted(records,key=lambda x:x['id']))
    print('HUMAN COMPLETE',len(records),sum(x['windows'] for x in records),flush=True)
if __name__=='__main__':main()
