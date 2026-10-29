"""Dense image-space human removal with MANO evidence and short-term flow.

Mask completion is recorded as estimated, separate from SAM image predictions.
"""
import argparse,json
from pathlib import Path
from collections import OrderedDict
import cv2,numpy as np
from human2robot.object_annotation.common import Progress,atomic_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--input',type=Path,default=Path('outputs/robot_composite_20261020/flow_full'));a=p.parse_args();a.root.mkdir(parents=True,exist_ok=True)
    cv2.setNumThreads(2);objects=Path('outputs/object_annotation_20261020');old=a.input/'human_masks'
    hands=Path('outputs/hand_mesh_repair_20261020/final');rows=[json.loads(l) for l in (hands/'frames.jsonl').read_text().splitlines()];meta=json.loads((objects/'objects.json').read_text());cache=OrderedDict()
    out=a.root/'human_masks';out.mkdir(exist_ok=True);evidence=a.root/'hand_pixels.npz'
    n=len(rows);points=np.full((n,2,21,2),np.nan,np.float32);masks=[];raws=[];progress=Progress(objects,'robot_repair_masks',n*2)
    try:
        for i,row in enumerate(rows):
            m=np.unpackbits(np.load(old/f'{i:06d}.npz')['mask'],count=640*480).reshape(480,640).astype(np.uint8)
            for h in row['hands']:
                file=hands/h['prediction_file'];key=str(file)
                if key not in cache:
                    cache[key]=dict(np.load(file))
                    if len(cache)>8:cache.popitem(last=False)
                saved=cache[key];j=h['prediction_index'];side=0 if h['handedness']=='Left' else 1
                xyz=saved['joints'][j]+saved['cam_t'][j];uv=xyz[:,:2]/xyz[:,2:]*12500+[320,240];points[i,side]=uv
                xyz=saved['vertices'][j]+saved['cam_t'][j];v=xyz[:,:2]/xyz[:,2:]*12500+[320,240]
                if np.isfinite(v).all() and (v.max(0)-v.min(0)).max()<220:
                    hull=cv2.convexHull(np.round(v).astype(np.int32));cv2.fillConvexPoly(m,hull,1)
            masks.append(m)
            if i%120==0:progress.update(i,phase='dense_MANO_mask_evidence')
        np.savez_compressed(evidence,joints_uv=points)
        flow=cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM);yy,xx=np.mgrid[:480,:640].astype(np.float32)
        def raw(i):
            s=meta['frames'][i]['chunk_start'];return cv2.imread(str(objects/'chunks'/f'{s:06d}'/f'{i-s:06d}.jpg'))
        def warp(m,prev,cur):
            f=flow.calc(cur,prev,None);return cv2.remap(m.astype(np.float32),xx+f[:,:,0],yy+f[:,:,1],cv2.INTER_LINEAR)
        # A short 3-frame union tolerates segmentation erosion and never carries
        # a body indefinitely across a new demonstration.
        forward=[];previous=None;carry=None
        for i in range(n):
            g=cv2.cvtColor(raw(i),cv2.COLOR_BGR2GRAY)
            if previous is None:carry=masks[i].astype(float)*3
            else:carry=np.maximum(masks[i]*3,np.maximum(0,warp(carry,previous,g)-1))
            forward.append((carry>0).astype(np.uint8));previous=g
        records=[];previous=None;carry=None
        for i in range(n-1,-1,-1):
            image=raw(i);g=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            if previous is None:carry=masks[i].astype(float)*3
            else:carry=np.maximum(masks[i]*3,np.maximum(0,warp(carry,previous,g)-1))
            m=(forward[i]>0)|(carry>0);m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((9,9),np.uint8));m=cv2.dilate(m,np.ones((9,9),np.uint8))>0
            np.savez_compressed(out/f'{i:06d}.npz',mask=np.packbits(m),frame_index=i)
            r=dict(frame_index=i,area=int(m.sum()),method='SAM3.1_anchor_flow_union_MANO_dense_and_adjacent_flow',accuracy_verified=False);records.append(r);atomic_json(out/f'{i:06d}.json',r)
            if i%120==0:
                progress.update(2*n-i,phase='bidirectional_mask_completion');v=image.copy();v[m]=(v[m]*.4+[25,25,150]).astype(np.uint8);cv2.imwrite(str(a.root/f'human_{i:06d}.jpg'),v)
            previous=g
        records.sort(key=lambda r:r['frame_index']);atomic_json(a.root/'human_mask_validation.json',dict(frames=records,summary=dict(total=n,method='SAM3.1+full-frame MANO+short-range flow',accuracy_verified=False)))
        progress.update(n*2,'completed')
    except Exception as e:progress.update(0,'failed',error=str(e));raise
    finally:progress.close()

if __name__=='__main__':main()
