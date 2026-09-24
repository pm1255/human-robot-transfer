"""RGB-only multi-person evidence view. Model handedness != verified anatomy."""
import argparse,json,subprocess,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

class HandTracker:
    def __init__(self):self.active={};self.next_id=0
    def update(self,hands,t):
        old=[(k,v) for k,v in self.active.items() if t-v['t']<.35]
        assigned={}
        if old and hands:
            cost=np.array([[np.linalg.norm(np.array(v['xy'])-h['xy'][0])+.04*(v['side']!=h['side']) for h in hands] for _,v in old])
            for i,j in zip(*linear_sum_assignment(cost)):
                if cost[i,j]<.25:assigned[int(j)]=old[i][0]
        for j,h in enumerate(hands):
            if j not in assigned:assigned[j]=self.next_id;self.next_id+=1
            k=assigned[j];h['track_id']=k;self.active[k]={'xy':h['xy'][0],'side':h['side'],'t':t}
        self.active={k:v for k,v in self.active.items() if t-v['t']<.35}
        return hands

def run(root,models,out):
    import cv2,mediapipe as mp
    root=Path(root);out=Path(out);out.mkdir(parents=True,exist_ok=True);models=Path(models)
    base=lambda name:mp.tasks.BaseOptions(model_asset_path=str(models/name),delegate=mp.tasks.BaseOptions.Delegate.CPU)
    V=mp.tasks.vision
    manifest=json.loads((root/'manifest.json').read_text());summaries=[]
    for src in manifest:
        dest=out/src['id'];dest.mkdir(exist_ok=True)
        if (dest/'summary.json').exists():summaries.append(json.loads((dest/'summary.json').read_text()));continue
        # Explicit resampling makes timestamps independent of decoder POS_MSEC conventions.
        proxy=dest/'proxy.avi'
        subprocess.run(['ffmpeg','-v','error','-y','-threads','1','-i',str(root/src['file']),'-t','12','-vf','scale=640:-2,fps=10','-an','-c:v','mjpeg','-threads','1','-q:v','3',str(proxy)],check=True)
        hand=V.HandLandmarker.create_from_options(V.HandLandmarkerOptions(base_options=base('hand_landmarker.task'),running_mode=V.RunningMode.VIDEO,num_hands=2,min_hand_detection_confidence=.3,min_hand_presence_confidence=.3,min_tracking_confidence=.3))
        pose=V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(base_options=base('pose_landmarker_full.task'),running_mode=V.RunningMode.VIDEO,num_poses=1,min_pose_detection_confidence=.4,min_pose_presence_confidence=.4,min_tracking_confidence=.4))
        obj=V.ObjectDetector.create_from_options(V.ObjectDetectorOptions(base_options=base('efficientdet_lite0.tflite'),running_mode=V.RunningMode.VIDEO,score_threshold=.25,max_results=12))
        cap=cv2.VideoCapture(str(proxy));tracker=HandTracker();records=[];images=[];features=[];masks=[];ids=[]
        while True:
            ok,bgr=cap.read()
            if not ok:break
            i=len(records);t=i/10;rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);im=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
            hh=hand.detect_for_video(im,i*100);pp=pose.detect_for_video(im,i*100);oo=obj.detect_for_video(im,i*100)
            hands=[]
            for j,points in enumerate(hh.hand_landmarks):
                hands.append({'xy':[[p.x,p.y] for p in points],'local_xyz':[[p.x,p.y,p.z] for p in hh.hand_world_landmarks[j]],'side':hh.handedness[j][0].category_name,'score':hh.handedness[j][0].score})
            tracker.update(hands,t)
            body=[];body3=[];full=False
            if pp.pose_landmarks:
                body=[[p.x,p.y,p.visibility,p.presence] for p in pp.pose_landmarks[0]]
                body3=[[p.x,p.y,p.z] for p in pp.pose_world_landmarks[0]]
                full=all(body[j][2]>.6 and body[j][3]>.6 and 0<=body[j][0]<=1 and 0<=body[j][1]<=1 for j in [11,12,23,24,25,26,27,28])
            objects=[]
            for d in oo.detections:
                c=d.categories[0];b=d.bounding_box
                objects.append({'label':c.category_name,'score':c.score,'box':[b.origin_x/rgb.shape[1],b.origin_y/rgb.shape[0],b.width/rgb.shape[1],b.height/rgb.shape[0]],'target_candidate':c.category_name.lower() in src.get('task','').lower()})
            # Slots are model Left/Right; duplicated label is ambiguous and excluded from supervision.
            f=np.zeros((2,3),np.float32);m=np.zeros(2,bool);track=np.full(2,-1,np.int64)
            for slot,side in enumerate(['Left','Right']):
                matches=[h for h in hands if h['side']==side]
                if len(matches)==1:
                    h=matches[0];xy=np.array(h['xy']);f[slot]=[*xy[[4,8]].mean(0),np.linalg.norm(xy[4]-xy[8])];m[slot]=True;track[slot]=h['track_id']
            records.append({'t':t,'hands':hands,'body':body,'body_local_xyz':body3,'full_body_visible':full,'objects':objects})
            images.append(cv2.resize(rgb,(128,128),interpolation=cv2.INTER_AREA));features.append(f);masks.append(m);ids.append(track)
        cap.release();hand.close();pose.close();obj.close();proxy.unlink()
        meta={**src,'frames':len(records),'both_hand_frames':sum(len(r['hands'])==2 for r in records),'any_hand_frames':sum(bool(r['hands']) for r in records),'full_body_frames':sum(r['full_body_visible'] for r in records),'object_frames':sum(bool(r['objects']) for r in records),'bottle_frames':sum(any(o['label']=='bottle' for o in r['objects']) for r in records),'mirror_status':'unknown','handedness_verified':False,'body_frame':'learned body-local, not world','timestamp_source':'ffmpeg uniform 10fps proxy','object_status':'COCO detections, target noun-match is weak label','robot_action_eligible':False,'model_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in models.iterdir() if p.is_file()}}
        (dest/'annotation.json').write_text(json.dumps({'meta':meta,'frames':records},separators=(',',':')))
        np.savez_compressed(dest/'supervision.npz',images=np.array(images),features=np.array(features),valid=np.array(masks),track_id=np.array(ids))
        (dest/'summary.json').write_text(json.dumps(meta,indent=2));summaries.append(meta);print(src['id'],meta['frames'],meta['both_hand_frames'],meta['full_body_frames'],flush=True)
    (out/'manifest.json').write_text(json.dumps(summaries,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root');p.add_argument('--models');p.add_argument('--out');a=p.parse_args();run(a.root,a.models,a.out)
