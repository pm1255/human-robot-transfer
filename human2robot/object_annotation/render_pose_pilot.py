"""Render observed cloth depth beside explicitly provisional FoundationPose poses."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from human2robot.hand_mesh.video import Writer, frames, probe
from .foundation_cloth import Masks
from .common import Progress,atomic_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();root=args.output
    data=json.loads((root/'pose6d.json').read_text())
    objects=json.loads((root/'objects.json').read_text())
    meta=objects['meta'];rows=data['frames'];h,w=meta['height'],meta['width']
    masks=Masks(root,(h,w))
    video_name='cloth_pose_review.mp4' if len(rows)==len(objects['frames']) else 'cloth_pose_pilot.mp4'
    progress=Progress(root,'geometry_render',len(rows))
    writer=Writer(root/video_name,w*2,h,meta['fps'])
    samples=[]
    sample_indices=set(np.linspace(30,len(rows)-1,min(18,len(rows)),dtype=int).tolist())
    sample_indices.update(i for i in [30,60,90,120,150,179] if i<len(rows))
    for i,bgr in enumerate(frames(meta['video'],probe(meta['video']),limit=len(rows))):
        record=rows[i];mask=masks.get(objects['frames'][i])
        with np.load(root/'depth'/f'{i:06d}.npz') as saved:
            depth=saved['depth'];K=saved['K']
        valid=mask&np.isfinite(depth)&(depth>0)
        left=bgr.copy();right=(bgr*.3).astype(np.uint8)
        if valid.any():
            lo,hi=np.percentile(depth[valid],[5,95])
            gray=(np.nan_to_num(np.clip((depth-lo)/max(float(hi-lo),1e-6),0,1))*255).astype(np.uint8)
            colour=cv2.applyColorMap(gray,cv2.COLORMAP_TURBO)
            right[valid]=colour[valid]
            cv2.putText(right,f'Model depth {lo:.2f}-{hi:.2f} m | scale unverified',(8,66),0,.45,(255,255,255),1,cv2.LINE_AA)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(left,contours,-1,(80,220,90),1)
        pose=record.get('T_camera_object')
        if pose is not None:
            pose=np.asarray(pose);length=max(.01,float(pose[2,3])*.06)
            axes=np.vstack((np.zeros(3),np.eye(3)*length))
            camera=axes@pose[:3,:3].T+pose[:3,3]
            pixels=camera@K.T;pixels=pixels[:,:2]/pixels[:,2:]
            if np.isfinite(pixels).all() and (camera[:,2]>0).all():
                pix=np.clip(pixels,-10000,10000).astype(int)
                for end,colour in zip(pix[1:],[(0,0,255),(0,255,0),(255,0,0)]):
                    cv2.line(left,tuple(pix[0]),tuple(end),colour,2,cv2.LINE_AA)
        for img,title in [(left,'FoundationPose | cloth global proxy'),(right,'MoGe-3 | deforming visible surface')]:
            cv2.rectangle(img,(0,0),(w,50),(25,25,25),-1)
            cv2.putText(img,title,(8,20),0,.5,(255,255,255),1,cv2.LINE_AA)
            cv2.putText(img,f'frame {i} | {record["status"]} | NOT GROUND TRUTH',(8,41),0,.43,(70,200,255),1,cv2.LINE_AA)
        combined=np.concatenate((left,right),axis=1);writer.write(combined)
        if i%120==0 or i==len(rows)-1:progress.update(i+1,phase='encoding')
        if i in sample_indices:
            cv2.imwrite(str(root/f'pose_review_{i:06d}.jpg'),combined);samples.append(i)
    writer.close()
    decoded=sum(1 for _ in frames(root/video_name,probe(root/video_name)))
    if decoded!=len(rows):raise RuntimeError('Incomplete pilot render')
    atomic_json(root/'pilot_video_validation.json',dict(video=video_name,decoded_frames=decoded,sample_frames=samples,
                depth_colours='per_frame_5th_to_95th_percentile_not_fixed_metric_colour_scale',
                axes='only_displayed_for_proxy_consistency_passes_still_review_required'))
    progress.update(decoded,'completed');progress.close()


if __name__=='__main__':main()
