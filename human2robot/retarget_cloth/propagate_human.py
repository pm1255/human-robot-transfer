"""Bidirectional optical flow between explicitly recorded SAM image anchors."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np
from human2robot.object_annotation.common import Progress,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--objects',type=Path,required=True);p.add_argument('--anchors',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--stride',type=int,default=30);a=p.parse_args()
    cv2.setNumThreads(2)
    data=json.loads((a.objects/'objects.json').read_text());n=len(data['frames']);out=a.root/'human_masks';out.mkdir(parents=True,exist_ok=True)
    anchors=sorted(set(range(0,n,a.stride))|{0,30,90,300,600,696,781,1200,2400,3600,3948,4800,n-1})
    monitor=Progress(a.objects,f'human_removal_flow_stride{a.stride}',n);monitor.update(0,'waiting_for_anchors');records=[]
    yy,xx=np.mgrid[:240,:320].astype(np.float32)
    flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    def rgb(i):
        s=data['frames'][i]['chunk_start'];return cv2.imread(str(a.objects/'chunks'/f'{s:06d}'/f'{i-s:06d}.jpg'))
    def mask(i):return np.unpackbits(np.load(a.anchors/'human_masks'/f'{i:06d}.npz')['mask'],count=640*480).reshape(480,640).astype(np.float32)
    def warp(m,previous,current):
        f=flow.calc(current,previous,None)
        return cv2.remap(m,xx+f[:,:,0],yy+f[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    def save(i,m,record):
        np.savez_compressed(out/f'{i:06d}.npz',mask=np.packbits(m),frame_index=i)
        record.update(frame_index=i,area=int(m.sum()),accuracy_verified=False);atomic_json(out/f'{i:06d}.json',record);records.append(record)
        if i%120==0:
            v=rgb(i);v[m]=(v[m]*.4+[25,25,150]).clip(0,255).astype(np.uint8);cv2.imwrite(str(a.root/f'human_{i:06d}.jpg'),v)
    try:
        for left,right in zip(anchors[:-1],anchors[1:]):
            while not (a.anchors/'human_masks'/f'{right:06d}.npz').exists():time.sleep(2)
            imgs=[cv2.resize(cv2.cvtColor(rgb(i),cv2.COLOR_BGR2GRAY),(320,240)) for i in range(left,right+1)]
            fwd=[cv2.resize(mask(left),(320,240),interpolation=cv2.INTER_AREA)]
            for k in range(1,len(imgs)):fwd.append(warp(fwd[-1],imgs[k-1],imgs[k]))
            back=[None]*len(imgs);back[-1]=cv2.resize(mask(right),(320,240),interpolation=cv2.INTER_AREA)
            for k in range(len(imgs)-2,-1,-1):back[k]=warp(back[k+1],imgs[k+1],imgs[k])
            for k,i in enumerate(range(left,right)):
                if k==0:save(i,mask(i)>.5,dict(method='SAM3.1_image_detector_CPU',anchor_frame=i))
                else:
                    t=k/(right-left);f,b=fwd[k],back[k];union=(f>.4)|(b>.4);intersection=(f>.4)&(b>.4);iou=float(intersection.sum()/max(1,union.sum()))
                    # Distance weighting avoids union ghosts from a distant anchor.
                    value=f*(1-t)+b*t
                    m=cv2.resize(value,(640,480),interpolation=cv2.INTER_LINEAR)>.38
                    m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8)).astype(bool)
                    save(i,m,dict(method='bidirectional_DIS_flow_estimate',anchors=[left,right],anchor_disagreement_iou=iou,review_required=iou<.65))
            monitor.update(right,phase='bidirectional_mask_propagation')
        save(n-1,mask(n-1)>.5,dict(method='SAM3.1_image_detector_CPU',anchor_frame=n-1))
        atomic_json(a.root/'human_mask_validation.json',dict(frames=records,summary=dict(total=n,model_inference_frames=len(anchors),flow_estimated_frames=n-len(anchors),review_frames=sum(r.get('review_required',False) for r in records),accuracy_verified=False)))
        monitor.update(n,'completed')
    except Exception as e:monitor.update(len(records),'failed',error=str(e));raise
    finally:monitor.close()

if __name__=='__main__':main()
